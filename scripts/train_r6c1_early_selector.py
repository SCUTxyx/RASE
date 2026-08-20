#!/usr/bin/env python3
"""R6-C.1C: train the early-window stratified risk selector.

Method (revised plan, six red lines):
  - Decisions only at t in {0, 8, 16}; after t16 the source is locked (no
    emergency trigger); switching at any decision point commits to persistent
    OFT to termination.
  - Two-stage controller: high-confidence danger at t0 switches immediately
    (no dwell); medium-risk groups wait for t8/t16 confirmation; high-confidence
    safe groups continue source.
  - Policy conditioning (seen VLA embedding and/or deployable behavior
    descriptor) + optional descriptor-conditioned FiLM adapter + direct
    success-advantage head.
  - 3-member task-bootstrap ensemble, 5 outer task-held-out folds, 5 training
  seeds.  Thresholds / LCB params are fit on an outer-train, task-held-out
    calibration split only; the final OOF uses each fold's own
    train-derived controller.  Same-task states/seeds/replicas never cross folds.

Supported ``--mode`` values (R6-C.2 ladder):
  per_vla        train on one VLA only (no policy conditioning)
  shared         shared core, no policy condition
  shared_id      shared core + VLA identity embedding
  shared_desc    shared core + deployable behavior descriptor
  shared_calib   shared core + descriptor-conditioned FiLM calibration
  loo            leave-one-VLA-out (descriptor from a few-shot calibration split)
  zero_shot      train on source VLA, eval on target VLA (challenge metric only)

Gate (per VLA, >=4/5 seeds): fold-correct success gap >= -5pp, original-protocol
false-continue <= 5%, absolute paired harm <= 5%, teacher-step savings >= 20%,
no concentrated suite harm.  Conditional missed-rescue is reported with a
point estimate and task-cluster interval only (no under-powered 5% hard gate).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.risk.light_risk_student import CandidateArmStudent  # noqa: E402
from rase.risk.tiny_universal_state_encoder import TinyUniversalStateEncoder  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folds(tasks: list[str], count: int, seed: int) -> list[set[str]]:
    values = sorted(set(tasks))
    random.Random(seed).shuffle(values)
    return [set(values[index::count]) for index in range(count)]


def calibration_tasks(train_tasks: set[str], fold: int, count: int = 6) -> set[str]:
    values = sorted(train_tasks)
    if not values:
        return set()
    offset = (fold * count) % len(values)
    rotated = values[offset:] + values[:offset]
    return set(rotated[: min(count, max(1, len(values) // 3))])


def normalize(values: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[fit].mean(0)
    std = values[fit].std(0)
    std[std < 1e-5] = 1.0
    return (values - mean) / std, mean, std


def eligible_masks(data: dict[str, np.ndarray], mode: str, target: str | None,
                   source: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, eval_mask).

    per_vla / shared* train on the target's (or all) tasks and evaluate the
    same rows; the per-VLA gate reads ``metrics_by_policy`` from the shared run.
    """
    policy = data["policy_id"]
    if mode == "per_vla":
        if not target:
            raise ValueError("per_vla requires --target-policy")
        return policy == target, policy == target
    if mode in ("shared", "shared_id", "shared_desc", "shared_calib"):
        # Shared core: train and evaluate on every VLA in the dataset.
        return np.ones(len(data["group_id"]), dtype=bool), np.ones(len(data["group_id"]), dtype=bool)
    if mode == "zero_shot":
        if not target or not source:
            raise ValueError("zero_shot requires target and source")
        return policy == source, policy == target
    if mode == "loo":
        if not target:
            raise ValueError("loo requires --target-policy")
        return policy != target, policy == target
    raise ValueError(mode)


def _descriptor_for(data: dict[str, np.ndarray], idx: np.ndarray,
                    policy: str) -> np.ndarray:
    """Compute a single VLA's behavior descriptor from the given rows."""
    mask = data["policy_id"][idx] == policy
    sub = idx[mask]
    if len(sub) == 0:
        raise ValueError(f"no rows for descriptor of {policy}")
    norm = np.linalg.norm
    return np.asarray([
        float(data["source_success"][sub].mean()),
        float(data["elapsed_source_steps"][sub].mean()),
        float(np.median(data["elapsed_source_steps"][sub])),
        float(norm(data["action_summary"][sub], axis=1).mean()),
        float(norm(data["proprio"][sub], axis=1).mean()),
        float(norm(data["history"][sub], axis=1).mean()),
        float(data["elapsed_progress"][sub].mean()),
        float(data["source_within_16"][sub].mean()),
    ], dtype=np.float32)


def policy_descriptors(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Deployable per-VLA behavior descriptor.

    Computed only from source-observable rollout statistics (never from OFT
    counterfactuals), so it can be produced for a new VLA from a handful of
    source rollouts.  Fields (8):
      source success rate, mean/median source steps, action norm,
      proprio norm, history norm, elapsed-progress mean, source-within-16 rate.
    """
    fields: dict[str, np.ndarray] = {}
    for policy in sorted(set(data["policy_id"][idx].tolist())):
        fields[policy] = _descriptor_for(data, idx, policy)
    return fields


def controller_early_window(data: dict[str, np.ndarray], idx: np.ndarray,
                            source_lcb: np.ndarray, advantage: np.ndarray,
                            risk_thr: float, adv_thr: float) -> dict:
    """Two-stage early-window controller (no emergency trigger).

    Decision points are t in {0, 8, 16} present in each group.  At t0 a
    high-risk, high-advantage row switches immediately; otherwise the group
    is observed until t8/t16 for the final judgment.  After t16 (or the last
    available decision point <= 16) the source is locked.  Baseline is
    ENTER_PERSISTENT_OFT at t0 (identical to R6-C).
    """
    if len(idx) != len(source_lcb) or len(idx) != len(advantage):
        raise ValueError("prediction arrays must align one-to-one with idx")
    # Keep both the global dataset row and its position in the prediction
    # arrays.  The previous implementation restarted ``position`` at zero for
    # every group and therefore reused the first group's scores for all groups.
    by_group: dict[str, list[tuple[int, int]]] = {}
    for local, row in enumerate(idx.tolist()):
        group = str(data["group_id"][row])
        by_group.setdefault(group, []).append((int(row), int(local)))

    entered = 0
    false_continue = 0
    missed_rescue = 0
    paired_harm = 0
    controller_success = 0
    controller_teacher = 0.0
    baseline_success = 0
    baseline_teacher = 0.0
    rescue_opportunities = 0
    n_trajectories = 0
    trajectory_records = []
    for group, members in by_group.items():
        members.sort(key=lambda item: int(data["elapsed_source_steps"][item[0]]))
        rows = [item[0] for item in members]
        prediction_positions = [item[1] for item in members]
        arm_success = data["arm_success"][rows]
        arm_steps = data["arm_teacher_steps"][rows]
        # Replica-aggregated labels are empirical probabilities.  Formal
        # episode metrics use the pre-registered majority adjudication; any
        # non-zero probability must not silently become True.
        persistent = arm_success[:, 1] > 0.5
        psteps = arm_steps[:, 1].astype(float)
        source_success = bool(arm_success[0, 0] > 0.5)
        baseline_success += int(persistent[0])
        baseline_teacher += float(psteps[0])
        rescue_opportunity = (not source_success) and bool(persistent[0])
        rescue_opportunities += int(rescue_opportunity)
        n_trajectories += 1

        enter_at = None
        enter_prediction_position = None
        for position, (row, prediction_position) in enumerate(zip(rows, prediction_positions)):
            elapsed = int(data["elapsed_source_steps"][row])
            if elapsed not in (0, 8, 16):
                continue
            risky = float(source_lcb[prediction_position]) < risk_thr
            worth = float(advantage[prediction_position]) > adv_thr
            if elapsed == 0:
                if risky and worth:
                    enter_at = position
                    enter_prediction_position = prediction_position
                    break
                # high-confidence safe / uncertain: observe until t8/t16.
                continue
            # t8 or t16: final judgment.
            if risky and worth:
                enter_at = position
                enter_prediction_position = prediction_position
                break
            if elapsed == 16:
                break  # lock source to termination.
        if enter_at is not None:
            entered += 1
            episode_success = bool(persistent[enter_at])
            episode_teacher = float(psteps[enter_at])
        else:
            episode_success = source_success
            episode_teacher = 0.0
            if rescue_opportunity:
                false_continue += 1
                missed_rescue += 1
        episode_paired_harm = bool(persistent[0]) and not episode_success
        paired_harm += int(episode_paired_harm)
        controller_success += int(episode_success)
        controller_teacher += episode_teacher
        trajectory_records.append({
            "group_id": group, "state_key": str(data["state_key"][rows[0]]),
            "task_id": str(data["task_id"][rows[0]]),
            "suite": str(data["suite"][rows[0]]),
            "policy_id": str(data["policy_id"][rows[0]]),
            "n_boundaries": len(rows),
            "entered_persistent": enter_at is not None,
            "source_success": bool(source_success),
            "persistent_success_at_0": bool(persistent[0]),
            "controller_success": bool(episode_success),
            "controller_teacher_steps": episode_teacher,
            "baseline_teacher_steps": float(psteps[0]),
            "baseline_success": bool(persistent[0]),
            "rescue_opportunity": bool(rescue_opportunity),
            "false_continue": bool(enter_at is None and rescue_opportunity),
            # Absolute paired harm is defined against the t0-persistent
            # baseline outcome, irrespective of whether failure came from no
            # entry or from entering after the rescue window had decayed.
            "paired_harm": bool(episode_paired_harm),
            "enter_elapsed_source_steps": (None if enter_at is None
                                             else int(data["elapsed_source_steps"][rows[enter_at]])),
            "enter_prediction_position": enter_prediction_position,
        })
    rescue_denom = max(1, baseline_success)
    n_denom = max(1, n_trajectories)
    return {
        "episodes": float(n_trajectories),
        "entered": float(entered),
        "successes": float(controller_success),
        "baseline_successes": float(baseline_success),
        "success_gap": float((controller_success - baseline_success) / n_denom),
        "false_continue": float(false_continue),
        "false_continue_rate": float(false_continue / rescue_denom),
        "missed_rescue": float(missed_rescue),
        "rescue_opportunities": float(rescue_opportunities),
        "paired_harm": float(paired_harm),
        "conditional_missed_rescue_rate": float(
            missed_rescue / max(1, rescue_opportunities)),
        "absolute_paired_harm": float(paired_harm / n_denom),
        "teacher_steps": float(controller_teacher),
        "baseline_teacher_steps": float(baseline_teacher),
        "savings": float(1.0 - controller_teacher / max(1.0, baseline_teacher)),
        "trajectories": trajectory_records,
    }


def select_controller(data: dict[str, np.ndarray], idx: np.ndarray,
                      source_lcb: np.ndarray, advantage: np.ndarray) -> tuple[dict, dict]:
    """Fit thresholds on an outer-train, task-held-out calibration split."""
    best = None
    lcb_values = sorted(set(float(v) for v in source_lcb.tolist()))
    risk_grid = sorted(set([0.01, 0.99, 1.01, *[q for q in np.linspace(0.05, 0.95, 19).tolist()],
                            *[v for v in lcb_values[:: max(1, len(lcb_values) // 12)]]]))
    adv_values = sorted(set(float(v) for v in advantage.tolist()))
    adv_grid = sorted(set([-1.0e9, -1.0, 1.0, 0.0, *[q for q in np.linspace(-0.5, 0.9, 15).tolist()],
                           *[v for v in adv_values[:: max(1, len(adv_values) // 12)]]]))
    for risk_thr in risk_grid:
        for adv_thr in adv_grid:
            metrics = controller_early_window(data, idx, source_lcb, advantage,
                                              risk_thr, adv_thr)
            if (metrics["success_gap"] < -0.05
                    or metrics["false_continue_rate"] > 0.05
                    or metrics["absolute_paired_harm"] > 0.05):
                continue
            rank = (metrics["savings"], metrics["success_gap"], -metrics["absolute_paired_harm"])
            if best is None or rank > best[0]:
                best = (rank, {"risk_thr": float(risk_thr), "adv_thr": float(adv_thr)}, metrics)
    if best is None:
        # Safety fallback is the frozen t0-persistent baseline, not an arbitrary
        # 0.5 threshold that can create unmeasured harm on the outer fold.
        params = {"risk_thr": 1.01, "adv_thr": -1.0e9}
        return params, controller_early_window(
            data, idx, source_lcb, advantage, params["risk_thr"], params["adv_thr"])
    return best[1], best[2]


def metrics_from_trajectory_records(records: list[dict]) -> dict:
    """Recompute controller metrics from independent trajectory records.

    This representation preserves multiplicity under a cluster bootstrap;
    resampling a task twice must count its trajectories twice rather than merge
    duplicate group IDs in the controller.
    """
    episodes = float(len(records))
    successes = float(sum(bool(r["controller_success"]) for r in records))
    baseline_successes = float(sum(bool(r["baseline_success"]) for r in records))
    false_continue = float(sum(bool(r["false_continue"]) for r in records))
    rescue_opportunities = float(sum(bool(r["rescue_opportunity"]) for r in records))
    # Recompute from outcomes so legacy records created before the corrected
    # definition cannot silently under-count late-entry harm.
    paired_harm = float(sum(
        bool(r["baseline_success"]) and not bool(r["controller_success"])
        for r in records
    ))
    teacher_steps = float(sum(float(r["controller_teacher_steps"]) for r in records))
    baseline_teacher_steps = float(sum(float(r["baseline_teacher_steps"]) for r in records))
    return {
        "episodes": episodes,
        "entered": float(sum(bool(r["entered_persistent"]) for r in records)),
        "successes": successes,
        "baseline_successes": baseline_successes,
        "success_gap": (successes - baseline_successes) / max(1.0, episodes),
        "false_continue": false_continue,
        "false_continue_rate": false_continue / max(1.0, baseline_successes),
        "missed_rescue": false_continue,
        "rescue_opportunities": rescue_opportunities,
        "conditional_missed_rescue_rate": false_continue / max(1.0, rescue_opportunities),
        "paired_harm": paired_harm,
        "absolute_paired_harm": paired_harm / max(1.0, episodes),
        "teacher_steps": teacher_steps,
        "baseline_teacher_steps": baseline_teacher_steps,
        "savings": 1.0 - teacher_steps / max(1.0, baseline_teacher_steps),
    }


def bootstrap_trajectory_records(records: list[dict], *, n_boot: int = 300,
                                 seed: int = 0) -> dict:
    """Task-cluster bootstrap over already frozen, fold-correct decisions."""
    rng = np.random.RandomState(seed)
    task_records: dict[str, list[dict]] = {}
    for record in records:
        task_records.setdefault(str(record["task_id"]), []).append(record)
    tasks = sorted(task_records)
    samples = {key: [] for key in
               ["success_gap", "false_continue_rate", "absolute_paired_harm", "savings",
                "conditional_missed_rescue_rate"]}
    for _ in range(n_boot):
        chosen = [rng.choice(tasks) for _ in tasks]
        boot_records = [record for task in chosen for record in task_records[task]]
        metrics = metrics_from_trajectory_records(boot_records)
        for key in samples:
            samples[key].append(metrics[key])
    result = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=float)
        result[key] = {"mean": float(arr.mean()),
                       "lower_95": float(np.quantile(arr, 0.025)),
                       "upper_95": float(np.quantile(arr, 0.975))}
    return result


def pooled_task_cluster_bootstrap(records: list[dict],
                                  n_boot: int = 300, seed: int = 0,
                                  policy: str | None = None) -> dict:
    """Bootstrap pooled OOF trajectory decisions without averaging thresholds."""
    selected = [record for record in records
                if policy is None or record["policy_id"] == policy]
    if not selected:
        return {key: {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0} for key in
                ["success_gap", "false_continue_rate", "absolute_paired_harm", "savings",
                 "conditional_missed_rescue_rate"]}
    return bootstrap_trajectory_records(selected, n_boot=n_boot, seed=seed)


def train_member(data: dict[str, np.ndarray], fit_idx: np.ndarray, *,
                 seed: int, epochs: int, device: str,
                 n_policies: int, policy_descriptor_dim: int,
                 use_calibration_adapter: bool, use_advantage_head: bool,
                 descriptors: dict[str, np.ndarray]) -> CandidateArmStudent:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    task_values = sorted(set(data["task_id"][fit_idx].tolist()))
    rng = random.Random(seed ^ 0xB0075A9)
    sampled = [rng.choice(task_values) for _ in task_values]
    boot = np.concatenate([fit_idx[data["task_id"][fit_idx] == task] for task in sampled])
    prop, _, _ = normalize(data["proprio"], boot)
    action, _, _ = normalize(data["action_summary"], boot)
    history, _, _ = normalize(data["history"], boot)
    progress, _, _ = normalize(data["elapsed_progress"].reshape(-1, 1), boot)

    encoder = TinyUniversalStateEncoder(
        image_size=96, proprio_dim=8, text_embed_dim=256, hidden_dim=128,
        output_dim=128, input_mode="image",
    )
    n_arms = int(data["arm_ids"].shape[0])
    model = CandidateArmStudent(
        encoder, n_arms=n_arms, proprio_dim=8,
        action_dim=data["action_summary"].shape[1],
        history_dim=data["history"].shape[1], fused_dim=128, head_hidden=128,
        n_members=1, n_cost_quantiles=3, use_unsafe_head=False, wm_dim=0,
        n_policies=n_policies, policy_emb_dim=16,
        policy_descriptor_dim=policy_descriptor_dim,
        use_calibration_adapter=use_calibration_adapter,
        use_advantage_head=use_advantage_head,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    image = torch.as_tensor(data["image"][boot].astype(np.float32) / 255.0, device=device)
    p = torch.as_tensor(prop[boot], device=device)
    a = torch.as_tensor(action[boot], device=device)
    h = torch.as_tensor(history[boot], device=device)
    lang = torch.as_tensor(data["language_hash"][boot], device=device)
    arm_succ = torch.as_tensor(data["arm_success"][boot], device=device)
    arm_successes = torch.as_tensor(data["arm_successes"][boot], device=device)
    arm_trials = torch.as_tensor(data["arm_trials"][boot], device=device)
    arm_cost_quantiles = torch.as_tensor(
        data["arm_teacher_step_quantiles"][boot], device=device
    )
    target_cost_quantiles = torch.log1p(arm_cost_quantiles)
    withins = torch.stack([
        torch.as_tensor(data["source_within_8"][boot], device=device),
        torch.as_tensor(data["source_within_16"][boot], device=device),
        torch.as_tensor(data["source_within_32"][boot], device=device),
    ], dim=-1)
    # Advantage target: success advantage of entering now vs continuing source.
    adv_target = arm_succ[:, 1] - arm_succ[:, 0]
    policy_index = None
    if n_policies > 0:
        policy_index = torch.as_tensor(data["policy_index"][boot], device=device)
    policy_desc = None
    if policy_descriptor_dim > 0:
        policy_desc = torch.as_tensor(
            np.stack([descriptors[str(data["policy_id"][i])] for i in boot]), device=device)

    for _ in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        out = model(image, p, a, h, lang, policy_index=policy_index,
                    policy_descriptor=policy_desc)
        source_loss = beta_binomial_nll(
            out["source_success"][0], out["arm_concentration"][0, :, 0],
            arm_successes[:, 0], arm_trials[:, 0],
        )
        within_loss = F.binary_cross_entropy(out["source_within"][0], withins)
        arm_loss = beta_binomial_nll(
            out["arm_success"][0], out["arm_concentration"][0],
            arm_successes, arm_trials,
        )
        # Replicas already define empirical q10/q50/q90 targets.  Fit those
        # directly rather than applying pinball loss to one median rollout.
        cost_loss = F.smooth_l1_loss(
            out["arm_cost"][0], target_cost_quantiles, beta=0.15
        )
        adv_loss = F.mse_loss(out["advantage"][0], adv_target)
        loss = (source_loss + 0.5 * within_loss + 0.5 * arm_loss
                + 0.12 * cost_loss + 0.5 * adv_loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    model.eval()
    model._r6_prop_mean = torch.as_tensor(data["proprio"][boot].mean(0), device=device)
    prop_std = data["proprio"][boot].std(0); prop_std[prop_std < 1e-5] = 1.0
    model._r6_prop_std = torch.as_tensor(prop_std, device=device)
    model._r6_action_mean = torch.as_tensor(data["action_summary"][boot].mean(0), device=device)
    action_std = data["action_summary"][boot].std(0); action_std[action_std < 1e-5] = 1.0
    model._r6_action_std = torch.as_tensor(action_std, device=device)
    model._r6_history_mean = torch.as_tensor(data["history"][boot].mean(0), device=device)
    history_std = data["history"][boot].std(0); history_std[history_std < 1e-5] = 1.0
    model._r6_history_std = torch.as_tensor(history_std, device=device)
    return model


def beta_binomial_nll(probability: torch.Tensor, concentration: torch.Tensor,
                      successes: torch.Tensor, trials: torch.Tensor) -> torch.Tensor:
    """Mean beta-binomial NLL for empirical replica counts.

    The binomial coefficient is retained so this remains a proper likelihood
    and comparable across rows with two versus three adjudication replicas.
    """
    probability = probability.clamp(1e-5, 1.0 - 1e-5)
    concentration = concentration.clamp(2.0, 200.0)
    alpha = probability * concentration
    beta = (1.0 - probability) * concentration
    failures = trials - successes
    log_choose = (
        torch.lgamma(trials + 1.0) - torch.lgamma(successes + 1.0)
        - torch.lgamma(failures + 1.0)
    )
    log_probability = (
        log_choose + torch.lgamma(successes + alpha)
        + torch.lgamma(failures + beta) - torch.lgamma(trials + alpha + beta)
        + torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
    )
    return -log_probability.mean()


@torch.no_grad()
def predict(model: CandidateArmStudent, data: dict[str, np.ndarray], idx: np.ndarray,
            device: str, descriptors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    image = torch.as_tensor(data["image"][idx].astype(np.float32) / 255.0, device=device)
    prop = (torch.as_tensor(data["proprio"][idx], device=device) - model._r6_prop_mean) / model._r6_prop_std
    action = (torch.as_tensor(data["action_summary"][idx], device=device) - model._r6_action_mean) / model._r6_action_std
    history = (torch.as_tensor(data["history"][idx], device=device) - model._r6_history_mean) / model._r6_history_std
    lang = torch.as_tensor(data["language_hash"][idx], device=device)
    policy_index = None
    if model.policy_embedding is not None:
        policy_index = torch.as_tensor(data["policy_index"][idx], device=device)
    policy_desc = None
    if model.descriptor_mlp is not None:
        policy_desc = torch.as_tensor(
            np.stack([descriptors[str(data["policy_id"][i])] for i in idx]), device=device)
    out = model(image, prop, action, history, lang, policy_index=policy_index,
                policy_descriptor=policy_desc)
    return {
        "source": out["source_success"].mean(0).cpu().numpy(),
        "within": out["source_within"].mean(0).cpu().numpy(),
        "arm_success": out["arm_success"].mean(0).cpu().numpy(),
        "arm_concentration": out["arm_concentration"].mean(0).cpu().numpy(),
        "arm_cost": torch.expm1(out["arm_cost"]).mean(0).cpu().numpy(),
        "advantage": out["advantage"].mean(0).cpu().numpy(),
    }


def fold_correct_aggregate(metrics_list: list[dict]) -> dict:
    """Sum episode-level counts across folds and recompute the derived rates.

    This is the fold-correct aggregation mandated by the plan (each fold's
    validation uses its own train-derived controller; averaging thresholds is
    explicitly forbidden).
    """
    counts = {
        "episodes": 0.0, "entered": 0.0, "successes": 0.0,
        "baseline_successes": 0.0, "false_continue": 0.0,
        "missed_rescue": 0.0, "rescue_opportunities": 0.0, "paired_harm": 0.0,
        "teacher_steps": 0.0, "baseline_teacher_steps": 0.0,
    }
    for metrics in metrics_list:
        for key in counts:
            counts[key] += float(metrics[key])
    episodes = max(1.0, counts["episodes"])
    baseline = max(1.0, counts["baseline_successes"])
    rescue_denom = max(1.0, counts["rescue_opportunities"])
    base_steps = max(1.0, counts["baseline_teacher_steps"])
    return {
        "episodes": counts["episodes"],
        "entered": counts["entered"],
        "successes": counts["successes"],
        "baseline_successes": counts["baseline_successes"],
        "success_gap": (counts["successes"] - counts["baseline_successes"]) / episodes,
        "false_continue": counts["false_continue"],
        "false_continue_rate": counts["false_continue"] / baseline,
        "rescue_opportunities": counts["rescue_opportunities"],
        "conditional_missed_rescue_rate": counts["missed_rescue"] / rescue_denom,
        "absolute_paired_harm": counts["paired_harm"] / episodes,
        "teacher_steps": counts["teacher_steps"],
        "baseline_teacher_steps": counts["baseline_teacher_steps"],
        "savings": 1.0 - counts["teacher_steps"] / base_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True,
                        choices=["per_vla", "shared", "shared_id", "shared_desc",
                                 "shared_calib", "loo", "zero_shot"])
    parser.add_argument("--target-policy")
    parser.add_argument("--source-policy")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260810)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lcb-z", type=float, default=1.6448536269514722)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    report = json.loads(args.dataset_report.read_text())
    if report.get("status") != "complete" or report.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("dataset/report lock mismatch")
    raw = np.load(args.dataset)
    data = {key: raw[key] for key in raw.files}
    required = {"arm_success", "arm_successes", "arm_trials",
                "arm_teacher_steps", "arm_teacher_step_quantiles"}
    if missing := sorted(required - set(data)):
        raise ValueError(f"candidate-arm dataset missing replica labels: {missing}")

    if report.get("protocol_sha256") != sha256(args.protocol):
        raise ValueError("dataset/protocol lock mismatch")
    train_mask, eval_mask = eligible_masks(data, args.mode, args.target_policy, args.source_policy)
    cohort_role = (data["cohort_role"] if "cohort_role" in data
                   else np.asarray(["natural"] * len(data["group_id"])))
    natural_mask = cohort_role == "natural"
    n_policies = int(data["policy_index"].max()) + 1
    policy_descriptor_dim = 0
    use_calibration_adapter = False
    use_advantage_head = True
    # No policy identity is allowed in per_vla/shared baselines.  Identity is
    # introduced only by the explicitly named shared_id ablation.
    n_policies_for_model = n_policies if args.mode == "shared_id" else 0
    if args.mode in ("shared_desc", "shared_calib"):
        policy_descriptor_dim = 8
    if args.mode == "shared_calib":
        use_calibration_adapter = True
    if args.mode == "loo":
        # New-VLA adaptation probe: shared core + descriptor only; the target
        # VLA's descriptor comes from a few-shot calibration split, never the
        # training folds.  No identity embedding for the held-out VLA.
        policy_descriptor_dim = 8
        n_policies_for_model = 0
    if args.mode == "zero_shot":
        # Challenge metric only (not a gate): the shared core is trained on the
        # source VLA with no identity embedding and no descriptor, so the target
        # evaluation measures pure cross-VLA transfer of the risk core alone.
        n_policies_for_model = 0

    # LOO few-shot calibration split: a small, task-disjoint subset of the
    # target VLA used ONLY to estimate its deployable behavior descriptor.
    loo_cal_rows: np.ndarray | None = None
    loo_target_descriptor: np.ndarray | None = None
    if args.mode == "loo":
        target_rows = np.where(eval_mask & natural_mask)[0]
        tasks = sorted(set(data["task_id"][target_rows].tolist()))
        random.Random(args.seed + 7).shuffle(tasks)
        cal_tasks = set(tasks[: max(1, len(tasks) // 4)])
        loo_cal_rows = np.where(eval_mask & np.isin(data["task_id"], list(cal_tasks)))[0]
        loo_target_descriptor = _descriptor_for(data, loo_cal_rows, args.target_policy)

    fold_task_rows = np.where(natural_mask & (train_mask | eval_mask))[0]
    task_folds = folds(data["task_id"][fold_task_rows].tolist(), args.folds, args.fold_seed)
    predictions: list[dict] = []
    fold_reports = []
    for fold, validation_tasks in enumerate(task_folds):
        train_tasks = set(data["task_id"].tolist()) - validation_tasks
        cal_tasks = calibration_tasks(train_tasks, fold)
        fit_tasks = train_tasks - cal_tasks
        fit_idx = np.where(train_mask & np.isin(data["task_id"], list(fit_tasks)))[0]
        # Enrichment is training-only.  Calibration and OOF validation use the
        # frozen natural cohort so hard-case oversampling cannot alter gates.
        cal_idx = np.where(train_mask & natural_mask
                           & np.isin(data["task_id"], list(cal_tasks)))[0]
        val_idx = np.where(eval_mask & natural_mask
                           & np.isin(data["task_id"], list(validation_tasks)))[0]
        if loo_cal_rows is not None:
            val_idx = np.asarray([i for i in val_idx if i not in set(loo_cal_rows.tolist())],
                                 dtype=int)
        if min(len(fit_idx), len(cal_idx), len(val_idx)) == 0:
            raise ValueError(f"fold {fold} has an empty partition")
        # Descriptors are computed from natural fit rows only, preventing the
        # enrichment mix from becoming an implicit policy identifier.
        descriptor_idx = fit_idx[natural_mask[fit_idx]]
        descriptors = policy_descriptors(data, descriptor_idx)
        if loo_target_descriptor is not None:
            descriptors[args.target_policy] = loo_target_descriptor
        models = [train_member(data, fit_idx,
                               seed=args.seed + fold * 1009 + member * 7919,
                               epochs=args.epochs, device=args.device,
                               n_policies=n_policies_for_model,
                               policy_descriptor_dim=policy_descriptor_dim,
                               use_calibration_adapter=use_calibration_adapter,
                               use_advantage_head=use_advantage_head,
                               descriptors=descriptors)
                  for member in range(args.members)]
        cal_pred = [predict(model, data, cal_idx, args.device, descriptors) for model in models]
        cal_source = np.stack([value["source"] for value in cal_pred])
        cal_lcb = np.clip(cal_source.mean(0) - args.lcb_z * cal_source.std(0), 0, 1)
        cal_adv_members = np.stack([value["advantage"] for value in cal_pred])
        cal_adv = (cal_adv_members.mean(0)
                   - args.lcb_z * cal_adv_members.std(0))
        params, _ = select_controller(data, cal_idx, cal_lcb, cal_adv)

        val_pred = [predict(model, data, val_idx, args.device, descriptors) for model in models]
        source = np.stack([value["source"] for value in val_pred])
        lcb = np.clip(source.mean(0) - args.lcb_z * source.std(0), 0, 1)
        adv_members = np.stack([value["advantage"] for value in val_pred])
        # ENTER requires a lower-confidence-bound advantage, not merely a high
        # ensemble mean.  This is the conservative counterpart to source LCB.
        adv = adv_members.mean(0) - args.lcb_z * adv_members.std(0)
        metrics = controller_early_window(data, val_idx, lcb, adv,
                                          params["risk_thr"], params["adv_thr"])
        bootstrap = bootstrap_trajectory_records(metrics["trajectories"],
                                                  seed=args.seed + fold)
        # Per-policy / per-suite metrics within this fold (each uses the fold's
        # own train-derived controller; aggregated fold-correctly later).
        fold_policy: dict[str, dict] = {}
        for policy in sorted(set(data["policy_id"][val_idx].tolist())):
            mask = data["policy_id"][val_idx] == policy
            fold_policy[policy] = {key: value for key, value in
                                   controller_early_window(data, val_idx[mask], lcb[mask], adv[mask],
                                                           params["risk_thr"], params["adv_thr"]).items()
                                   if key != "trajectories"}
        fold_suite: dict[str, dict] = {}
        for suite in sorted(set(data["suite"][val_idx].tolist())):
            mask = data["suite"][val_idx] == suite
            fold_suite[suite] = {key: value for key, value in
                                 controller_early_window(data, val_idx[mask], lcb[mask], adv[mask],
                                                         params["risk_thr"], params["adv_thr"]).items()
                                 if key != "trajectories"}
        # Per-policy-per-suite metrics: the per-VLA gate must check suite
        # concentration within each policy, not pooled across policies.
        fold_policy_suite: dict[str, dict[str, dict]] = {}
        for policy in sorted(set(data["policy_id"][val_idx].tolist())):
            policy_mask = data["policy_id"][val_idx] == policy
            for suite in sorted(set(data["suite"][val_idx][policy_mask].tolist())):
                mask = policy_mask & (data["suite"][val_idx] == suite)
                fold_policy_suite.setdefault(policy, {})[suite] = {
                    key: value for key, value in
                    controller_early_window(data, val_idx[mask], lcb[mask], adv[mask],
                                            params["risk_thr"], params["adv_thr"]).items()
                    if key != "trajectories"}
        fold_reports.append({
            "fold": fold, "controller_params": params,
            "fit_rows": len(fit_idx), "calibration_rows": len(cal_idx),
            "validation_rows": len(val_idx),
            "validation_metrics": {key: value for key, value in metrics.items()
                                   if key != "trajectories"},
            "validation_metrics_by_policy": fold_policy,
            "validation_metrics_by_suite": fold_suite,
            "validation_metrics_by_policy_suite": fold_policy_suite,
            "validation_bootstrap": bootstrap,
            "trajectories": metrics["trajectories"],
        })
        for local, index in enumerate(val_idx):
            predictions.append({
                "index": int(index), "fold": fold,
                "group_id": str(data["group_id"][index]),
                "state_key": str(data["state_key"][index]),
                "task_id": str(data["task_id"][index]),
                "suite": str(data["suite"][index]),
                "policy_id": str(data["policy_id"][index]),
                "elapsed_source_steps": int(data["elapsed_source_steps"][index]),
                "source_success": bool(data["source_success"][index]),
                "persistent_success": bool(data["arm_success"][index, 1] > 0.5),
                "persistent_empirical_probability": float(data["arm_success"][index, 1]),
                "persistent_successes": int(data["arm_successes"][index, 1]),
                "persistent_trials": int(data["arm_trials"][index, 1]),
                "source_mean": float(source[:, local].mean()),
                "source_std": float(source[:, local].std()),
                "source_lcb": float(lcb[local]),
                "advantage": float(adv[local]),
                "advantage_lcb": float(adv[local]),
                "advantage_mean": float(adv_members[:, local].mean()),
                "advantage_std": float(adv_members[:, local].std()),
                "risk_thr": float(params["risk_thr"]),
                "adv_thr": float(params["adv_thr"]),
                "risky": bool(lcb[local] < params["risk_thr"]),
                "persistent_success_mean": float(np.stack(
                    [value["arm_success"][local, 1] for value in val_pred]).mean()),
                "persistent_success_std": float(np.stack(
                    [value["arm_success"][local, 1] for value in val_pred]).std()),
                "persistent_concentration_mean": float(np.stack(
                    [value["arm_concentration"][local, 1] for value in val_pred]).mean()),
                "persistent_cost_q50": float(np.stack(
                    [value["arm_cost"][local, 1, 1] for value in val_pred]).mean()),
                "persistent_cost_q90": float(np.stack(
                    [value["arm_cost"][local, 1, 2] for value in val_pred]).mean()),
            })
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions.sort(key=lambda row: (row["group_id"], row["elapsed_source_steps"]))
    # Fold-correct aggregation: each fold's validation metrics were produced by
    # that fold's own train-derived controller.  Averaging thresholds and
    # re-evaluating all OOF rows is forbidden (avg-threshold error).
    metrics = fold_correct_aggregate([report["validation_metrics"] for report in fold_reports])
    by_policy = {}
    for policy in sorted({name for report in fold_reports for name in report["validation_metrics_by_policy"]}):
        by_policy[policy] = fold_correct_aggregate([
            report["validation_metrics_by_policy"][policy]
            for report in fold_reports if policy in report["validation_metrics_by_policy"]])
    by_suite = {}
    for suite in sorted({name for report in fold_reports for name in report["validation_metrics_by_suite"]}):
        by_suite[suite] = fold_correct_aggregate([
            report["validation_metrics_by_suite"][suite]
            for report in fold_reports if suite in report["validation_metrics_by_suite"]])
    by_policy_suite = {}
    for policy in sorted({name for report in fold_reports
                          for name in report["validation_metrics_by_policy_suite"]}):
        by_policy_suite[policy] = {}
        for suite in sorted({name for report in fold_reports
                             for name in report["validation_metrics_by_policy_suite"].get(policy, {})}):
            by_policy_suite[policy][suite] = fold_correct_aggregate([
                report["validation_metrics_by_policy_suite"][policy][suite]
                for report in fold_reports
                if policy in report["validation_metrics_by_policy_suite"]
                and suite in report["validation_metrics_by_policy_suite"][policy]])
    # Task-cluster bootstrap aggregated across folds (resample tasks within the
    # pooled OOF rows, but evaluate each row with its own fold's controller).
    # Shared* reports keep the pooled bootstrap AND per-policy bootstraps so the
    # per-VLA interval never pools policies.
    trajectory_records = [record for report in fold_reports for record in report["trajectories"]]
    bootstrap = pooled_task_cluster_bootstrap(trajectory_records, seed=args.seed)
    bootstrap_by_policy = {
        policy: pooled_task_cluster_bootstrap(trajectory_records, seed=args.seed + 11,
                                              policy=policy)
        for policy in by_policy
    }

    def _gate(m, by_suite_metrics):
        return (m["success_gap"] >= -0.05 and m["false_continue_rate"] <= 0.05
                and m["absolute_paired_harm"] <= 0.05 and m["savings"] >= 0.20
                and all(v["success_gap"] >= -0.05 and v["absolute_paired_harm"] <= 0.05
                        for v in by_suite_metrics.values()))
    gate = _gate(metrics, by_suite)
    # Per-policy gate: the per-VLA stage gate uses each policy's own fold-correct
    # metrics and its own suite concentration (never pooled across policies).
    gate_by_policy = {
        policy: _gate(m, by_policy_suite.get(policy, {}))
        for policy, m in by_policy.items()
    }
    trajectory_records.sort(key=lambda row: (row["group_id"], row["controller_teacher_steps"]))
    result = {
        "schema_version": "rase-r6c1-early-selector-oof/v1",
        "status": "complete",
        "scientific_scope": "policy-conditioned early-window t={0,8,16} stratified selector; "
                            "no emergency trigger; outer-train task-held-out calibration; "
                            "enrichment-free natural eval",
        "dataset": str(args.dataset.resolve()), "dataset_sha256": sha256(args.dataset),
        "protocol": str(args.protocol.resolve()), "protocol_sha256": sha256(args.protocol),
        "mode": args.mode, "target_policy": args.target_policy, "source_policy": args.source_policy,
        "seed": args.seed, "fold_seed": args.fold_seed, "folds": args.folds,
        "members": args.members, "epochs": args.epochs, "lcb_z": args.lcb_z,
        "policy_descriptor_dim": policy_descriptor_dim,
        "use_calibration_adapter": use_calibration_adapter,
        "metrics": {key: value for key, value in metrics.items() if key != "trajectories"},
        "metrics_bootstrap": bootstrap,
        "metrics_bootstrap_by_policy": bootstrap_by_policy,
        "metrics_by_policy": by_policy,
        "metrics_by_suite": by_suite,
        "metrics_by_policy_suite": by_policy_suite,
        "gate_by_policy": gate_by_policy,
        "seed_gate_passed": gate,
        "fold_reports": fold_reports,
        "trajectory_records": trajectory_records,
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in
                      ["mode", "target_policy", "seed", "metrics", "metrics_by_suite",
                       "seed_gate_passed"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
