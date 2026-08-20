#!/usr/bin/env python3
"""Paired evaluation runner for Route C Plugin.

Supports strict counterfactual pairing:
  - B0: pure student rollout
  - B1d: student until stagnation, then OFT teacher takeover
  - B3: student until stagnation, then student + Plugin takeover
  - B3 force-no-takeover: B3 arm that never activates the plugin (parity check)
  - parity: runs B0 + B3-force-off in one process, saves separate jsonl files

Uses episode manifest to ensure identical (task, init_state, seed) across arms.
Outputs jsonl with per-episode results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rase.collect.libero_env_factory import make_libero_env_for_task
from rase.collect.forked_rollout import load_smolvla_policy_bundle
from rase.collect.oracle_continuation import OracleChunkContinuation
from rase.collect.policy_step import as_batched_action, select_env_action, success_from_info
from rase.collect.pool_candidates import observation_from_libero_env
from rase.oracle.client import OracleClient


# ------------------------------------------------------------------ helpers

def _import_torch():
    global torch
    import torch


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    _import_torch()
    torch.manual_seed(seed)


def _progress(control_env: Any) -> float:
    try:
        pos = getattr(control_env.env, "_eef_xpos", None)
        if pos is not None:
            return float(np.linalg.norm(np.asarray(pos)))
    except Exception:
        pass
    return 0.0


def _proprio(obs: dict) -> np.ndarray:
    p = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).flatten()
    q = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32).flatten()
    out = np.concatenate([p, q])[:7]
    return out.astype(np.float32)


def _make_stagnation_detector(plugin_conf: dict):
    """Create a standalone stagnation detector using the same params as plugin."""
    from rase.recovery.stagnation import StagnationDetector
    return StagnationDetector(
        window=plugin_conf["stagnation_window"],
        eps=plugin_conf["stagnation_eps"],
    )


# ------------------------------------------------------------------ arms

def run_b0(handle, bundle, instruction, max_steps, *, trace: bool = False) -> dict:
    obs = observation_from_libero_env(handle.vector_env.envs[0])
    frames = []
    for t in range(max_steps):
        action = select_env_action(bundle, obs, task=instruction)
        if trace:
            frames.append({"t": t, "action": action.flatten().tolist()[:7],
                           "progress": 0.0, "done": False})
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            if trace and frames:
                frames[-1]["done"] = True
                frames[-1]["success"] = success_from_info(info)
            return {"success": success_from_info(info), "steps": t + 1, "arm": "B0",
                    "trace": frames if trace else None}
    return {"success": False, "steps": max_steps, "arm": "B0",
            "trace": frames if trace else None}


def run_b1d(handle, bundle, client, instruction, max_student, max_teacher,
            protocol: dict, *, trace: bool = False) -> dict:
    obs = observation_from_libero_env(handle.vector_env.envs[0])
    progress_vals = []
    frames = []
    boundary_t = -1
    stagnation_eps = protocol["plugin_config"]["stagnation_eps"]
    stagnation_window = protocol["plugin_config"]["stagnation_window"]

    for t in range(max_student):
        action = select_env_action(bundle, obs, task=instruction)
        if trace:
            frames.append({"t": t, "action": action.flatten().tolist()[:7],
                           "progress": 0.0, "done": False, "phase": "student"})
        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            return {"success": success_from_info(info), "steps": t + 1, "arm": "B1d",
                    "handover": False, "trace": frames if trace else None}
        progress_vals.append(_progress(handle.control_env))
        if t >= stagnation_window and np.std(progress_vals[-stagnation_window:]) < stagnation_eps:
            boundary_t = t + 1
            break

    if boundary_t >= 0:
        oft = OracleChunkContinuation(client, instruction=instruction, control_env=handle.control_env)
        for ti in range(max_teacher):
            t_action = oft.act(obs, task=instruction)
            if trace:
                frames.append({"t": boundary_t + ti, "action": t_action.flatten().tolist()[:7],
                               "progress": 0.0, "done": False, "phase": "oft"})
            obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(t_action))
            obs = observation_from_libero_env(handle.vector_env.envs[0])
            terminated = bool(np.asarray(term).reshape(-1)[0])
            truncated = bool(np.asarray(trunc).reshape(-1)[0])
            if terminated or truncated:
                return {"success": success_from_info(info), "steps": boundary_t + ti + 1,
                        "arm": "B1d", "handover": True, "boundary_step": boundary_t,
                        "trace": frames if trace else None}
        return {"success": False, "steps": boundary_t + max_teacher, "arm": "B1d",
                "handover": True, "boundary_step": boundary_t,
                "trace": frames if trace else None}
    return {"success": False, "steps": max_student, "arm": "B1d", "handover": False,
            "trace": frames if trace else None}


def _build_lean_obs_features(proprio, student_action, stag_len, prog_delta):
    """Build 16-D lean obs features: proprio(7)+action(7)+stats(2). No latent."""
    p = np.asarray(proprio, dtype=np.float32).flatten()[:7]
    a = np.asarray(student_action, dtype=np.float32).flatten()[:7]
    p_pad = np.zeros(7, dtype=np.float32)
    a_pad = np.zeros(7, dtype=np.float32)
    p_pad[:len(p)] = p
    a_pad[:len(a)] = a
    # This field is aligned context length (max 8), not true stagnation time.
    # Keep it bit-for-bit consistent with collect_activation_labels.py.
    stag = np.array([min(stag_len / 8.0, 1.0), np.clip(prog_delta, -1.0, 1.0)], dtype=np.float32)
    return np.concatenate([p_pad, a_pad, stag]).astype(np.float32)


def _safety_clip_action(mixed, student_action, last_action,
                         ACTION_RATE_LIMIT=0.1, SAFE_ACTION_RATE_LIMIT=0.3):
    """Apply per-step action rate limit (both normal and hard safety cap)."""
    if last_action is not None:
        mixed = np.clip(mixed, last_action - ACTION_RATE_LIMIT,
                        last_action + ACTION_RATE_LIMIT)
        # Hard safety cap: never exceed SAFE_ACTION_RATE_LIMIT
        mixed = np.clip(mixed, last_action - SAFE_ACTION_RATE_LIMIT,
                        last_action + SAFE_ACTION_RATE_LIMIT)
    return mixed


def run_b3(handle, bundle, executor, instruction, max_steps,
           feature_extractor=None, feature_level="F2",
           force_no_takeover=False, selector=None, plugin=None,
           gate=None, use_lean_features=False, gate_threshold=0.5,
           constant_delta=None, delta_scale=1.0,
           *, trace=False) -> dict:
    """Run B3. If selector+plugin are provided, use selector-based triggering
    instead of the stagnation executor. If gate+plugin are provided, use the
    trained activation gate for binary on/off decisions."""
    _import_torch()
    obs = observation_from_libero_env(handle.vector_env.envs[0])
    if executor is not None:
        executor.reset()
    frames = []
    takeover_count = 0
    takeover_action_steps = 0
    gate_queries = 0
    gate_positive_decisions = 0
    use_selector = (selector is not None and plugin is not None)
    use_gate = (gate is not None and (plugin is not None or constant_delta is not None))

    # Selector/gate-mode state
    selector_history = []        # list of (proprio, student_action, progress)
    selector_taking_over = False
    selector_consecutive_student = 0  # counter for handback
    selector_last_action = None
    selector_cooldown_counter = 0     # steps remaining in cooldown
    ACTION_DIM = 7
    PROPRIO_DIM = 8
    HISTORY_WINDOW = 8
    HAND_BACK_WINDOW = 3          # consecutive "no" to hand back
    MAX_TAKEOVER_STEPS = 8        # hard cap: force re-evaluation
    TAKEOVER_COOLDOWN = 4         # steps before selector can re-trigger
    ACTION_RATE_LIMIT = 0.1
    SAFE_ACTION_RATE_LIMIT = 0.3  # hard safety cap
    DELTA_CLIP = 0.5
    MIX_RAMP = [0.0, 0.3, 0.6, 1.0]

    # Safety envelope state
    post_recovery_counter = 0       # consecutive progress improvements after takeover
    prev_progress_for_recovery = 0.0
    in_recovery_grace = False        # disable F0 after recovery
    RECOVERY_DISABLE_N = 3           # consecutive improvements to disable

    from rase.collect.smolvla_feature_extractor import build_feature_vector

    for t in range(max_steps):
        gate_prob = None
        # ── Capture SmolVLA latent from SAME forward pass as select_env_action ──
        if not force_no_takeover and feature_extractor is not None and feature_level == "F2":
            feature_extractor.start_capture()

        student_action = select_env_action(bundle, obs, task=instruction)

        if not force_no_takeover and feature_extractor is not None and feature_level == "F2":
            smolvla_latent = feature_extractor.finish_capture()
        else:
            smolvla_latent = None

        progress_val = _progress(handle.control_env)
        proprio_val = _proprio(obs)

        if force_no_takeover:
            action = student_action
            phase = "student_forced"
        elif use_selector:
            # --- Selector v2: delta-aware + bounded takeover ---
            obs_feat = build_feature_vector(
                smolvla_latent=smolvla_latent,
                proprio=proprio_val,
                student_action=student_action.flatten(),
                stagnation_length=len(selector_history),
                progress_delta=0.0,
                feature_level=feature_level,
            )

            # Build history tensor
            hist_arr = np.zeros((HISTORY_WINDOW, 8 + 7 + 1 + 7), dtype=np.float32)
            recent = selector_history[-HISTORY_WINDOW:]
            for hi, h in enumerate(recent):
                p = np.asarray(h["proprio"], dtype=np.float32).flatten()
                a = np.asarray(h["student_action"], dtype=np.float32).flatten()
                p_pad = np.zeros(PROPRIO_DIM, dtype=np.float32)
                a_pad = np.zeros(ACTION_DIM, dtype=np.float32)
                p_pad[:min(len(p), PROPRIO_DIM)] = p[:PROPRIO_DIM]
                a_pad[:min(len(a), ACTION_DIM)] = a[:ACTION_DIM]
                idx = hi + HISTORY_WINDOW - len(recent)
                if idx >= 0:
                    hist_arr[idx] = np.concatenate([p_pad, a_pad, [float(h["progress"])], a_pad])

            # Match obs_feat to plugin's expected input dim (F0 may differ from F2)
            obs_feat_plugin = obs_feat
            if hasattr(plugin, 'obs_feature_dim') and len(obs_feat_plugin) != plugin.obs_feature_dim:
                obs_feat_plugin = np.zeros(plugin.obs_feature_dim, dtype=np.float32)
                n_copy = min(len(obs_feat), plugin.obs_feature_dim)
                obs_feat_plugin[:n_copy] = obs_feat[:n_copy]

            # Compute plugin delta BEFORE asking selector (Phase 3)
            delta = plugin.predict_delta(hist_arr, obs_feat_plugin, student_action.flatten())
            delta_clipped = np.clip(delta, -DELTA_CLIP, DELTA_CLIP)
            delta_norm = float(np.linalg.norm(delta_clipped))

            # --- Take-over/handback decision (Phase 4: bounded takeover) ---
            if selector_cooldown_counter > 0:
                # In cooldown: force student, decrement counter
                selector_cooldown_counter -= 1
                should_use = False
            else:
                should_use = selector.should_intervene(
                    hist_arr, obs_feat, student_action.flatten(),
                    plugin_delta=delta_clipped, delta_norm=delta_norm,
                    threshold=0.4)

            # Hard cap: if we have been taking over too long, force handback
            takeover_duration = len([1 for h in selector_history[-MAX_TAKEOVER_STEPS:]
                                     if h.get("phase") == "takeover"]) if selector_taking_over else 0
            if selector_taking_over and takeover_duration >= MAX_TAKEOVER_STEPS:
                selector_taking_over = False
                selector_consecutive_student = 0
                selector_cooldown_counter = TAKEOVER_COOLDOWN

            if selector_taking_over:
                if not should_use:
                    selector_consecutive_student += 1
                    if selector_consecutive_student >= HAND_BACK_WINDOW:
                        selector_taking_over = False
                        selector_consecutive_student = 0
                        selector_cooldown_counter = TAKEOVER_COOLDOWN
                else:
                    selector_consecutive_student = 0
            else:
                if should_use and selector_cooldown_counter <= 0:
                    selector_taking_over = True
                    takeover_count += 1
                    selector_consecutive_student = 0

            if selector_taking_over:
                g = MIX_RAMP[min(takeover_count, len(MIX_RAMP) - 1)]
                mixed = np.clip(student_action.flatten() + g * delta_clipped, -1.0, 1.0)
                if selector_last_action is not None:
                    mixed = np.clip(mixed, selector_last_action - ACTION_RATE_LIMIT,
                                    selector_last_action + ACTION_RATE_LIMIT)
                action = mixed.reshape(1, -1)
                selector_last_action = action.flatten().copy()
                phase = "selector_takeover"
            else:
                action = student_action
                phase = "student"
                selector_last_action = None

            selector_history.append({
                "proprio": proprio_val,
                "student_action": student_action.flatten(),
                "progress": float(progress_val),
                "phase": "takeover" if phase == "selector_takeover" else "student",
            })
            if len(selector_history) > HISTORY_WINDOW * 2:
                selector_history = selector_history[-HISTORY_WINDOW * 2:]
        elif use_gate:
            # --- Trained binary activation gate with safety envelope ---
            context_len = min(len(selector_history), HISTORY_WINDOW)
            progress_delta = 0.0
            if selector_history:
                progress_delta = float(progress_val - selector_history[-1]["progress"])
            if use_lean_features:
                obs_feat_gate = _build_lean_obs_features(
                    proprio_val, student_action.flatten(),
                    context_len, progress_delta)
            else:
                obs_feat_gate = build_feature_vector(
                    smolvla_latent=smolvla_latent,
                    proprio=proprio_val,
                    student_action=student_action.flatten(),
                    stagnation_length=context_len,
                    progress_delta=progress_delta,
                    feature_level=feature_level,
                )

            # Build history tensor
            hist_arr = np.zeros((HISTORY_WINDOW, 8 + 7 + 1 + 7), dtype=np.float32)
            recent = selector_history[-HISTORY_WINDOW:]
            for hi, h in enumerate(recent):
                p = np.asarray(h["proprio"], dtype=np.float32).flatten()
                a = np.asarray(h["student_action"], dtype=np.float32).flatten()
                p_pad = np.zeros(PROPRIO_DIM, dtype=np.float32)
                a_pad = np.zeros(ACTION_DIM, dtype=np.float32)
                p_pad[:min(len(p), PROPRIO_DIM)] = p[:PROPRIO_DIM]
                a_pad[:min(len(a), ACTION_DIM)] = a[:ACTION_DIM]
                idx = hi + HISTORY_WINDOW - len(recent)
                if idx >= 0:
                    hist_arr[idx] = np.concatenate([p_pad, a_pad, [float(h["progress"])], a_pad])

            if constant_delta is not None:
                delta = np.asarray(constant_delta, dtype=np.float32) * float(delta_scale)
            else:
                obs_feat_plugin = obs_feat_gate
                if hasattr(plugin, 'obs_feature_dim') and len(obs_feat_plugin) != plugin.obs_feature_dim:
                    obs_feat_plugin = np.zeros(plugin.obs_feature_dim, dtype=np.float32)
                    n_copy = min(len(obs_feat_gate), plugin.obs_feature_dim)
                    obs_feat_plugin[:n_copy] = obs_feat_gate[:n_copy]
                delta = plugin.predict_delta(hist_arr, obs_feat_plugin, student_action.flatten())
            delta_clipped = np.clip(delta, -DELTA_CLIP, DELTA_CLIP)
            delta_norm = float(np.linalg.norm(delta_clipped))

            # Gate decision
            if selector_cooldown_counter > 0:
                selector_cooldown_counter -= 1
                should_use = False
            elif in_recovery_grace:
                should_use = False
            else:
                gate_prob = gate.predict(
                    hist_arr, obs_feat_gate.astype(np.float32),
                    student_action.flatten().astype(np.float32),
                    delta_clipped.astype(np.float32), float(delta_norm))
                gate_queries += 1
                should_use = gate_prob >= gate_threshold
                if should_use:
                    gate_positive_decisions += 1

            # Hard cap + handback
            takeover_duration = len([1 for h in selector_history[-MAX_TAKEOVER_STEPS:]
                                     if h.get("phase") == "takeover"]) if selector_taking_over else 0
            if selector_taking_over and takeover_duration >= MAX_TAKEOVER_STEPS:
                selector_taking_over = False
                selector_consecutive_student = 0
                selector_cooldown_counter = TAKEOVER_COOLDOWN

            if selector_taking_over:
                if not should_use:
                    selector_consecutive_student += 1
                    if selector_consecutive_student >= HAND_BACK_WINDOW:
                        selector_taking_over = False
                        selector_consecutive_student = 0
                        selector_cooldown_counter = TAKEOVER_COOLDOWN
                        in_recovery_grace = True
                else:
                    selector_consecutive_student = 0
            else:
                if should_use and selector_cooldown_counter <= 0 and not in_recovery_grace:
                    selector_taking_over = True
                    takeover_count += 1
                    selector_consecutive_student = 0

            if selector_taking_over:
                g = MIX_RAMP[min(takeover_count, len(MIX_RAMP) - 1)]
                mixed = np.clip(student_action.flatten() + g * delta_clipped, -1.0, 1.0)
                mixed = _safety_clip_action(mixed, student_action.flatten(),
                                             selector_last_action,
                                             ACTION_RATE_LIMIT, SAFE_ACTION_RATE_LIMIT)
                action = mixed.reshape(1, -1)
                selector_last_action = action.flatten().copy()
                phase = "gate_takeover"
            else:
                action = student_action
                phase = "student"
                selector_last_action = None

            selector_history.append({
                "proprio": proprio_val,
                "student_action": student_action.flatten(),
                "progress": float(progress_val),
                "phase": "takeover" if phase == "gate_takeover" else "student",
            })
            if len(selector_history) > HISTORY_WINDOW * 2:
                selector_history = selector_history[-HISTORY_WINDOW * 2:]

            # Post-recovery detection
            if not selector_taking_over:
                if progress_val > prev_progress_for_recovery + 0.01:
                    post_recovery_counter += 1
                else:
                    post_recovery_counter = 0
                prev_progress_for_recovery = progress_val
                if post_recovery_counter >= RECOVERY_DISABLE_N:
                    in_recovery_grace = False
                    post_recovery_counter = 0
        elif executor is not None:
            stal_len = len(executor._stagnation.progress_values) if hasattr(executor, '_stagnation') else 0
            obs_feat = build_feature_vector(
                smolvla_latent=smolvla_latent,
                proprio=proprio_val,
                student_action=student_action.flatten(),
                stagnation_length=stal_len,
                progress_delta=0.0,
                feature_level=feature_level,
            )

            action, info = executor.step(obs, student_action, progress_val, obs_features=obs_feat)
            executor.record_history(proprio_val, student_action, progress_val, obs_feat)

            if executor.check_takeover_after_update():
                takeover_count += 1
            phase = info.get("phase", "student")
            if info.get("takeover"):
                takeover_count += 1
        else:
            action = student_action
            phase = "student"

        if phase in ("gate_takeover", "selector_takeover", "takeover", "plugin_takeover"):
            takeover_action_steps += 1

        if trace:
            frames.append({"t": t, "action": action.flatten().tolist()[:7],
                           "progress": float(progress_val), "done": False,
                           "phase": phase, "gate_prob": gate_prob})

        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            return {"success": success_from_info(info), "steps": t + 1, "arm": "B3",
                    "takeover_steps": takeover_action_steps,
                    "takeover_entries": takeover_count,
                    "gate_queries": gate_queries,
                    "gate_positive_decisions": gate_positive_decisions,
                    "gate_threshold": gate_threshold,
                    "force_no_takeover": force_no_takeover,
                    "trace": frames if trace else None}
    return {"success": False, "steps": max_steps, "arm": "B3",
            "takeover_steps": takeover_action_steps,
            "takeover_entries": takeover_count,
            "gate_queries": gate_queries,
            "gate_positive_decisions": gate_positive_decisions,
            "gate_threshold": gate_threshold,
            "force_no_takeover": force_no_takeover,
            "trace": frames if trace else None}


def run_b3_always_on(handle, bundle, plugin, instruction, max_steps,
                     feature_extractor=None, feature_level="F2",
                     constant_delta=None, delta_scale=1.0,
                     *, trace=False) -> dict:
    """Always-On mode: plugin delta applied at every step, no handback, no bounded takeover."""
    _import_torch()
    obs = observation_from_libero_env(handle.vector_env.envs[0])
    frames = []
    takeover_steps = 0

    ACTION_DIM = 7
    PROPRIO_DIM = 8
    from rase.collect.smolvla_feature_extractor import build_feature_vector

    # History buffer for plugin conditioning
    plugin_history = []
    HISTORY_WINDOW = 8

    for t in range(max_steps):
        if feature_extractor is not None and feature_level == "F2":
            feature_extractor.start_capture()

        student_action = select_env_action(bundle, obs, task=instruction)

        if feature_extractor is not None and feature_level == "F2":
            smolvla_latent = feature_extractor.finish_capture()
        else:
            smolvla_latent = None

        progress_val = _progress(handle.control_env)
        proprio_val = _proprio(obs)

        # Build obs features
        obs_feat = build_feature_vector(
            smolvla_latent=smolvla_latent,
            proprio=proprio_val,
            student_action=student_action.flatten(),
            stagnation_length=len(plugin_history),
            progress_delta=0.0,
            feature_level=feature_level,
        )

        # Compute delta: constant or from plugin
        if constant_delta is not None:
            delta = np.array(constant_delta, dtype=np.float32) * delta_scale
        else:
            # Build history tensor
            hist_arr = np.zeros((HISTORY_WINDOW, 8 + 7 + 1 + 7), dtype=np.float32)
            recent = plugin_history[-HISTORY_WINDOW:]
            for hi, h in enumerate(recent):
                p = np.asarray(h["proprio"], dtype=np.float32).flatten()
                a = np.asarray(h["student_action"], dtype=np.float32).flatten()
                p_pad = np.zeros(PROPRIO_DIM, dtype=np.float32)
                a_pad = np.zeros(ACTION_DIM, dtype=np.float32)
                p_pad[:min(len(p), PROPRIO_DIM)] = p[:PROPRIO_DIM]
                a_pad[:min(len(a), ACTION_DIM)] = a[:ACTION_DIM]
                idx = hi + HISTORY_WINDOW - len(recent)
                if idx >= 0:
                    hist_arr[idx] = np.concatenate([p_pad, a_pad, [float(h["progress"])], a_pad])

            obs_feat_plugin = obs_feat
            if hasattr(plugin, 'obs_feature_dim') and len(obs_feat_plugin) != plugin.obs_feature_dim:
                obs_feat_plugin = np.zeros(plugin.obs_feature_dim, dtype=np.float32)
                n_copy = min(len(obs_feat), plugin.obs_feature_dim)
                obs_feat_plugin[:n_copy] = obs_feat[:n_copy]

            delta = plugin.predict_delta(hist_arr, obs_feat_plugin,
                                          student_action.flatten()) * delta_scale

        delta_clipped = np.clip(delta, -0.5, 0.5)
        mixed = np.clip(student_action.flatten() + delta_clipped, -1.0, 1.0)
        action = mixed.reshape(1, -1)

        takeover_steps += 1
        phase = "always_on"

        plugin_history.append({
            "proprio": proprio_val,
            "student_action": student_action.flatten(),
            "progress": float(progress_val),
        })
        if len(plugin_history) > HISTORY_WINDOW * 2:
            plugin_history = plugin_history[-HISTORY_WINDOW * 2:]

        if trace:
            frames.append({"t": t, "action": action.flatten().tolist()[:7],
                           "progress": float(progress_val), "done": False,
                           "phase": phase})

        obs, reward, term, trunc, info = handle.vector_env.step(as_batched_action(action))
        obs = observation_from_libero_env(handle.vector_env.envs[0])
        terminated = bool(np.asarray(term).reshape(-1)[0])
        truncated = bool(np.asarray(trunc).reshape(-1)[0])
        if terminated or truncated:
            return {"success": success_from_info(info), "steps": t + 1, "arm": "B3_AO",
                    "takeover_steps": takeover_steps,
                    "trace": frames if trace else None}
    return {"success": False, "steps": max_steps, "arm": "B3_AO",
            "takeover_steps": takeover_steps,
            "trace": frames if trace else None}


# ------------------------------------------------------------------ parity runner

def run_parity(protocol: dict, bundle: dict, manifest: list[dict],
               output_dir: Path, max_steps: int) -> dict:
    """Run B0 + B3-force-off pairs in the same process. Returns parity metrics."""
    b0_results = []
    b3_results = []
    for i, entry in enumerate(manifest):
        task_id = entry["task_id"]
        seed_val = entry["seed"]
        init_state = entry["init_state_id"] % 50
        suite = entry.get("suite", "libero_spatial")

        # ── B0 (with trace) ──
        bundle["policy"].reset()
        seed_everything(seed_val)
        handle = make_libero_env_for_task(task_id, init_state_id=init_state,
                                            seed=seed_val, libero_flavor="clean")
        instruction = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")
        r0 = run_b0(handle, bundle, instruction, max_steps, trace=True)
        handle.close()
        # Reset policy action queue so next env starts fresh
        bundle["policy"].reset()

        r0.update({"suite": suite, "task_id": task_id, "seed": seed_val,
                    "init_state_id": init_state, "arm": "b0"})
        b0_results.append(r0)

        # Ensure CUDA + EGL cleanup before recreating env
        import time
        _import_torch()
        torch.cuda.synchronize()
        time.sleep(3.0)

        # ── B3 force-no-takeover (same seed, with trace) ──
        seed_everything(seed_val)
        handle = make_libero_env_for_task(task_id, init_state_id=init_state,
                                            seed=seed_val, libero_flavor="clean")
        instruction = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")
        # No plugin, no extractor — pure B0 path
        r3 = run_b3(handle, bundle, None, instruction, max_steps,
                    force_no_takeover=True, trace=True)
        handle.close()
        r3.update({"suite": suite, "task_id": task_id, "seed": seed_val,
                    "init_state_id": init_state, "arm": "b3"})
        b3_results.append(r3)

        # Compare per-step actions
        trace0 = r0.get("trace", [])
        trace3 = r3.get("trace", [])
        n_common = min(len(trace0), len(trace3))
        action_mismatches = 0
        max_action_l2 = 0.0
        for t in range(n_common):
            a0 = np.array(trace0[t]["action"], dtype=np.float32)
            a3 = np.array(trace3[t]["action"], dtype=np.float32)
            l2 = float(np.linalg.norm(a0 - a3))
            max_action_l2 = max(max_action_l2, l2)
            if l2 > 1e-6:
                action_mismatches += 1

        action_ok = action_mismatches == 0
        success_ok = r0["success"] == r3["success"]
        print(f"  [{i+1}/{len(manifest)}] {task_id} s={seed_val} "
              f"B0: {r0['success']} ({r0['steps']})  B3(off): {r3['success']} ({r3['steps']})  "
              f"action_ok={action_ok} success_ok={success_ok} "
              f"(action_mismatches={action_mismatches}/{n_common} "
              f"max_action_l2={max_action_l2:.2e})")

    # Save
    b0_path = output_dir / "parity_b0.jsonl"
    b3_path = output_dir / "parity_b3.jsonl"
    for path, results in [(b0_path, b0_results), (b3_path, b3_results)]:
        with open(path, "w") as f:
            for r in results:
                r_clean = {k: v for k, v in r.items() if k != "trace"}
                f.write(json.dumps(r_clean) + "\n")

    # Compare: per-step action parity is the primary gate
    action_matches = 0
    success_matches = 0
    total_actions = 0
    total_action_mismatches = 0
    max_action_l2 = 0.0
    step_diffs = []
    for a, b in zip(b0_results, b3_results):
        trace0 = a.get("trace", [])
        trace3 = b.get("trace", [])
        n = min(len(trace0), len(trace3))
        pair_ok = True
        pair_mismatches = 0
        pair_max_l2 = 0.0
        for t in range(n):
            a0_v = np.array(trace0[t]["action"], dtype=np.float32)
            a3_v = np.array(trace3[t]["action"], dtype=np.float32)
            l2 = float(np.linalg.norm(a0_v - a3_v))
            pair_max_l2 = max(pair_max_l2, l2)
            total_actions += 1
            if l2 > 1e-6:
                pair_ok = False
                pair_mismatches += 1
                total_action_mismatches += 1
        max_action_l2 = max(max_action_l2, pair_max_l2)
        if pair_ok:
            action_matches += 1
        if a["success"] == b["success"]:
            success_matches += 1
        step_diffs.append(abs(a["steps"] - b["steps"]))

    n_pairs = len(b0_results)
    parity = {
        "n_pairs": n_pairs,
        "action_match": action_matches,
        "action_match_rate": action_matches / n_pairs if n_pairs else 0,
        "total_action_mismatches": total_action_mismatches,
        "total_actions_compared": total_actions,
        "max_action_l2": float(max_action_l2),
        "success_match": success_matches,
        "success_match_rate": success_matches / n_pairs if n_pairs else 0,
        "max_step_diff": max(step_diffs),
        "mean_step_diff": float(np.mean(step_diffs)),
        # Gate: all per-step actions must be identical (code path parity)
        # Success is secondary (LIBERO environment has inherent non-determinism)
        "action_parity_pass": action_matches == n_pairs,
        "pass": action_matches == n_pairs,
        "details": [
            {
                "task_id": a["task_id"],
                "seed": a["seed"],
                "init_state_id": a["init_state_id"],
                "b0_success": a["success"],
                "b0_steps": a["steps"],
                "b3_success": b["success"],
                "b3_steps": b["steps"],
                "action_match": True,  # will be populated below
            }
            for a, b in zip(b0_results, b3_results)
        ],
    }

    parity_path = output_dir / "no_takeover_parity.json"
    parity_path.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    print(f"\nParity result: action_match={action_matches}/{n_pairs} = {action_matches/n_pairs:.2%} "
          f"({'PASS' if parity['pass'] else 'FAIL'})")
    print(f"  success_match={success_matches}/{n_pairs} (secondary)")
    print(f"  total_action_mismatches={total_action_mismatches}/{total_actions} "
          f"max_action_l2={max_action_l2:.2e}")
    print(f"Saved to: {parity_path}, {b0_path}, {b3_path}")
    return parity


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plugin-ckpt", type=Path, default=None)
    parser.add_argument("--selector-ckpt", type=Path, default=None,
                        help="selector checkpoint for selector-based B3 takeover")
    parser.add_argument("--gate-ckpt", type=Path, default=None,
                        help="trained binary activation gate checkpoint (LeanActivationGate)")
    parser.add_argument("--gate-threshold", type=float, default=None,
                        help="override the threshold stored in the gate checkpoint")
    parser.add_argument("--lean-features", action="store_true",
                        help="use lean 16-D obs features (no SmolVLA latent) for gate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, default=None)
    parser.add_argument("--arm", type=str, default="b3",
                        choices=["b0", "b1d", "b3", "parity"])
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--max-student-steps", type=int, default=300)
    parser.add_argument("--max-teacher-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--oft-server-port", type=int, default=5555)
    parser.add_argument("--feature-level", type=str, default="F2",
                        choices=["F0", "F1", "F2"])
    parser.add_argument("--force-no-takeover", action="store_true")
    parser.add_argument("--parity-manifest", type=Path, default=None,
                        help="manifest for B0/B3 parity pairs")
    parser.add_argument("--parity-b0-manifest", type=Path, default=None,
                        help="B0-only manifest for parity")
    parser.add_argument("--parity-b3-manifest", type=Path, default=None,
                        help="B3-only manifest for parity")
    parser.add_argument("--trace-jsonl", type=Path, default=None)
    parser.add_argument("--always-on", action="store_true",
                        help="Always-On mode: plugin active at every step, no handback")
    parser.add_argument("--constant-delta", type=Path, default=None,
                        help="JSON file with f0_constant_vector_c for constant delta mode")
    parser.add_argument("--delta-scale", type=float, default=1.0,
                        help="scale factor for delta (default 1.0)")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    plugin_conf = protocol["plugin_config"]

    # Load SmolVLA
    policy_path = Path(protocol["student_identity"]["checkpoint_path"])
    vlm_cache = protocol.get("vlm_cache_path", "")
    bundle = load_smolvla_policy_bundle(
        policy_path, device="cuda",
        tokenizer_path=vlm_cache if vlm_cache else None,
        observation_height=360, observation_width=360,
    )

    # ── Parity mode ──
    if args.arm == "parity":
        if args.parity_manifest:
            manifest = json.loads(args.parity_manifest.read_text(encoding="utf-8"))
            # Use only b0 entries for pairing (we'll run both arms per entry)
            manifest = [m for m in manifest if m.get("arm") == "b0"]
        else:
            # Generate simple manifest
            task_ids = protocol["splits"][args.suite]["dev"]
            if args.task_id:
                task_ids = [args.task_id]
            manifest = []
            for task_id in task_ids[:2]:
                for ep_i in range(args.n_episodes):
                    seed_val = (args.seed * 31 + ep_i * 7) % (2**31)
                    init_state = (args.init_state_id + ep_i) % 50
                    manifest.append({
                        "task_id": task_id,
                        "init_state_id": init_state,
                        "seed": seed_val,
                        "arm": "b0",
                        "suite": args.suite,
                    })

        parity = run_parity(protocol, bundle, manifest, output_dir, args.max_student_steps)
        return 0 if parity["pass"] else 1

    # ── Regular arms ──

    # Plugin for B3
    plugin = None
    executor = None
    feature_extractor = None
    selector = None
    gate = None
    gate_threshold = 0.5
    constant_delta = None
    if args.plugin_ckpt and args.arm in ("b3",) and not args.force_no_takeover:
        from rase.recovery.residual_plugin import load_plugin
        from rase.recovery.plugin_executor import RecoveryPluginExecutor
        from rase.collect.smolvla_feature_extractor import SmolVLAFeatureExtractor

        plugin = load_plugin(str(args.plugin_ckpt))
        plugin.eval()
        if args.feature_level == "F2":
            feature_extractor = SmolVLAFeatureExtractor(bundle)

        if args.constant_delta and args.constant_delta.is_file():
            cd_data = json.loads(args.constant_delta.read_text(encoding="utf-8"))
            constant_delta = cd_data.get("f0_constant_vector_c", None)
            print(f"Constant delta loaded: {args.constant_delta}")

        if not args.always_on:
            # Load gate if provided (binary activation gate mode)
            if args.gate_ckpt and args.gate_ckpt.is_file():
                _import_torch()
                ckpt = torch.load(str(args.gate_ckpt), map_location="cpu")
                from scripts.train_activation_gate import LeanActivationGate
                cfg = ckpt["config"]
                gate = LeanActivationGate(**cfg)
                gate.load_state_dict(ckpt["model_state_dict"])
                gate.eval()
                gate = gate.to("cuda")
                gate_threshold = float(ckpt.get("activation_threshold", 0.5))
                if args.gate_threshold is not None:
                    gate_threshold = float(args.gate_threshold)
                print(f"Gate loaded: {args.gate_ckpt} threshold={gate_threshold:.3f}")
            # Load selector if provided (selector-based takeover mode)
            elif constant_delta is None and args.selector_ckpt and args.selector_ckpt.is_file():
                from rase.recovery.selector import load_selector
                selector = load_selector(str(args.selector_ckpt))
                selector.eval()
                print(f"Selector loaded: {args.selector_ckpt}")
            elif constant_delta is None:
                # Fall back to stagnation-based executor
                executor = RecoveryPluginExecutor(
                    plugin, bundle,
                    history_window=plugin_conf["plugin_history_window"],
                    stagnation_window=plugin_conf["stagnation_window"],
                    stagnation_eps=plugin_conf["stagnation_eps"],
                    max_takeover_steps=plugin_conf["max_takeover_steps"],
                    delta_clip=plugin_conf["delta_clip_per_dim"],
                )

    # OFT client for B1d
    client = OracleClient(f"tcp://127.0.0.1:{args.oft_server_port}", timeout_ms=60000) \
        if args.arm == "b1d" else None

    # Build episode list
    if args.episode_manifest:
        manifest = json.loads(args.episode_manifest.read_text(encoding="utf-8"))
        manifest = [m for m in manifest if m.get("arm", "b3") == args.arm]
    else:
        task_ids = protocol["splits"][args.suite]["dev"]
        if args.task_id:
            task_ids = [args.task_id]
        manifest = []
        for task_id in task_ids[:2]:
            for ep_i in range(args.n_episodes):
                seed_val = (args.seed * 31 + ep_i * 7) % (2**31)
                init_state = (args.init_state_id + ep_i) % 50
                manifest.append({
                    "task_id": task_id,
                    "init_state_id": init_state,
                    "seed": seed_val,
                    "arm": args.arm,
                    "suite": args.suite,
                })

    print(f"Running {len(manifest)} episodes (arm={args.arm}, "
          f"force_no_takeover={args.force_no_takeover})")

    results = []
    trace_entries = []
    for i, entry in enumerate(manifest):
        task_id = entry["task_id"]
        seed_val = entry["seed"]
        init_state = entry["init_state_id"] % 50
        suite = entry.get("suite", args.suite)

        seed_everything(seed_val)
        bundle["policy"].reset()
        handle = make_libero_env_for_task(task_id, init_state_id=init_state,
                                            seed=seed_val, libero_flavor="clean")
        instruction = str(getattr(handle.vector_env.envs[0], "task_description", "") or "")

        arm = entry.get("arm", args.arm)

        if arm == "b0":
            r = run_b0(handle, bundle, instruction, args.max_student_steps,
                       trace=bool(args.trace_jsonl))
        elif arm == "b1d" and client:
            r = run_b1d(handle, bundle, client, instruction,
                        args.max_student_steps, args.max_teacher_steps,
                        protocol, trace=bool(args.trace_jsonl))
        elif arm == "b3":
            if args.always_on:
                r = run_b3_always_on(handle, bundle, plugin, instruction,
                                     args.max_student_steps,
                                     feature_extractor=feature_extractor,
                                     feature_level=args.feature_level,
                                     constant_delta=constant_delta,
                                     delta_scale=args.delta_scale,
                                     trace=bool(args.trace_jsonl))
            else:
                r = run_b3(handle, bundle, executor, instruction, args.max_student_steps,
                           feature_extractor=feature_extractor,
                           feature_level=args.feature_level,
                           force_no_takeover=args.force_no_takeover,
                           selector=selector, plugin=plugin,
                           gate=gate, use_lean_features=args.lean_features,
                           gate_threshold=gate_threshold,
                           constant_delta=constant_delta,
                           delta_scale=args.delta_scale,
                           trace=bool(args.trace_jsonl))
        else:
            r = None

        handle.close()

        if r is not None:
            r.update({"suite": suite, "task_id": task_id, "seed": seed_val,
                       "init_state_id": init_state, "arm": arm})
            results.append(r)
            if args.trace_jsonl and r.get("trace"):
                trace_entry = {k: v for k, v in r.items() if k != "trace"}
                trace_entry["trace"] = r["trace"]
                trace_entries.append(trace_entry)
            extra = f"steps={r['steps']}"
            if arm == "b3":
                extra += f" takeover={r.get('takeover_steps', '?')}"
            print(f"  [{i+1}/{len(manifest)}] {task_id} s={seed_val} {arm}: "
                  f"success={r['success']} {extra}")

    if client:
        client.close()

    # Save
    suffix = f"_{args.arm}"
    if args.force_no_takeover:
        suffix += "_notakeover"
    results_path = output_dir / f"paired_results{suffix}.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            r_clean = {k: v for k, v in r.items() if k != "trace"}
            f.write(json.dumps(r_clean) + "\n")

    if args.trace_jsonl and trace_entries:
        trace_path = args.trace_jsonl
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trace_path, "w") as f:
            for t in trace_entries:
                f.write(json.dumps(t) + "\n")
        print(f"Traces saved to: {trace_path}")

    n_success = sum(1 for r in results if r.get("success"))
    print(f"\n=== {args.arm} Summary ===")
    print(f"  N={len(results)}  Success: {n_success}/{len(results)} "
          f"= {n_success/len(results):.2%}")
    print(f"Results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
