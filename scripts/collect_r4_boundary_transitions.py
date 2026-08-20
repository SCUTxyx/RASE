#!/usr/bin/env python3
"""Collect exact-state, action-conditioned PRE-C0-R4 boundary transitions.

For each PRE-A3 train state, follow the persistent OFT trajectory.  At frozen
handoff boundaries h={0,8,16,32,64,96,128}, fork the simulator for one Student
step and one OFT step, then record deterministic SmolVLA latents and both next
latents.  Outcome labels are joined from the completed cost-aware PRE-A3 audit;
simulator restore is used only for counterfactual supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        value = yaml.safe_load(text)
    else:
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _expand(value: object | None, env_name: str | None = None) -> str | None:
    if value in {None, ""}:
        return os.environ.get(env_name) if env_name else None
    return str(Path(os.path.expandvars(str(value))).expanduser())


def _operator_for_boundary(h: int) -> str:
    return "CONTINUE" if h == 0 else f"OFT_H{h}"


def _success_from_step(term: Any, trunc: Any, info: Any) -> tuple[bool, bool, bool]:
    from rase.collect.policy_step import success_from_info

    terminated = bool(np.asarray(term).reshape(-1)[0])
    truncated = bool(np.asarray(trunc).reshape(-1)[0])
    return terminated, truncated, bool(success_from_info(info)) if (terminated or truncated) else False


@contextmanager
def preserve_rng_state() -> Iterable[None]:
    """Keep feature queries from perturbing simulator or rollout randomness."""
    import torch

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class StableLatentCapture:
    """Capture expert hidden states with a deterministic frozen projection."""

    def __init__(self, bundle: Mapping[str, Any], *, latent_dim: int, seed: int) -> None:
        import torch
        import torch.nn.functional as functional

        self.model = bundle["policy"].model
        layer = self.model.action_out_proj
        self.hidden_dim = int(getattr(layer, "in_features", 864))
        self.latent_dim = int(latent_dim)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        weight = torch.randn(self.latent_dim, self.hidden_dim, generator=generator)
        self.weight_cpu = functional.normalize(weight, dim=1).contiguous()
        device = next(bundle["policy"].parameters()).device
        self.weight = self.weight_cpu.to(device=device)
        self.projection_sha256 = hashlib.sha256(
            self.weight_cpu.numpy().tobytes()
        ).hexdigest()
        self._captured = None
        self._handle = None

    def _hook(self, _module: Any, args: tuple[Any, ...]) -> None:
        import torch

        if args and isinstance(args[0], torch.Tensor):
            self._captured = args[0].detach()

    def start(self) -> None:
        self.close()
        self._captured = None
        self._handle = self.model.action_out_proj.register_forward_pre_hook(self._hook)

    def finish(self) -> np.ndarray:
        import torch.nn.functional as functional

        self.close()
        if self._captured is None:
            raise RuntimeError("SmolVLA action_out_proj hook captured no hidden state")
        hidden = self._captured
        pooled = hidden.mean(dim=1) if hidden.ndim == 3 else hidden
        latent = functional.linear(pooled, self.weight)
        return latent.squeeze(0).float().cpu().numpy().astype(np.float32)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def query_student(
    bundle: Mapping[str, Any],
    capture: StableLatentCapture,
    observation: Mapping[str, Any],
    *,
    task: str,
    seed: int,
    chunk_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    from rase.collect.forked_rollout import InProcessSmolVLAContinuation

    student = InProcessSmolVLAContinuation(bundle, temperature=0.5, seed=seed)
    with preserve_rng_state():
        student.reset()
        capture.start()
        try:
            first_action = np.asarray(
                student.act(observation, task=task), dtype=np.float32
            ).reshape(-1, 7)[0]
            actions = [first_action]
            latent = capture.finish()
        except Exception:
            capture.close()
            raise
        for _ in range(1, int(chunk_steps)):
            actions.append(
                np.asarray(student.act(observation, task=task), dtype=np.float32)
                .reshape(-1, 7)[0]
            )
    return latent, np.stack(actions)


def rollout_student_handback(
    *,
    branch_restored: Any,
    boundary_snapshot: Any,
    bundle: Mapping[str, Any],
    task: str,
    seed: int,
) -> dict[str, Any]:
    """Label handback by running Student to termination from this exact boundary."""
    from rase.collect.forked_rollout import InProcessSmolVLAContinuation
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    with preserve_rng_state():
        branch_restored.forkable.restore(
            boundary_snapshot,
            check_task_fingerprint=branch_restored.check_task_fingerprint,
        )
        single = branch_restored.handle.vector_env.envs[0]
        vector_env = branch_restored.handle.vector_env
        horizon = int(getattr(single, "_max_episode_steps", 600))
        observation = observation_from_libero_env(single)
        student = InProcessSmolVLAContinuation(bundle, temperature=0.5, seed=seed)
        student.reset()
        steps = 0
        while True:
            if current_timestep(branch_restored.handle.control_env) >= horizon:
                return {
                    "success": False,
                    "steps": steps,
                    "stop_reason": "horizon",
                }
            action = np.asarray(student.act(observation, task=task), dtype=np.float32)
            observation, _, term, trunc, info = vector_env.step(
                as_batched_action(action)
            )
            steps += 1
            terminated, truncated, success = _success_from_step(term, trunc, info)
            if terminated or truncated:
                return {
                    "success": bool(success),
                    "steps": steps,
                    "stop_reason": (
                        "success" if success else (
                            "terminated" if terminated else "truncated"
                        )
                    ),
                }


def _labels_by_state(audit: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if audit.get("safe_handback_status") != "ready":
        raise ValueError(
            "safe-handback gate is closed: " + repr(audit.get("safe_handback_reasons"))
        )
    labels = {}
    for row in audit.get("per_state", []):
        labels[str(row["state_key"])] = dict(row)
    return labels


def _minimum_successful_steps(label: Mapping[str, Any]) -> int | None:
    pairs = []
    for operator, success in label.get("operator_success", {}).items():
        if not success or not str(operator).startswith("OFT_H"):
            continue
        pairs.append((int(str(operator).removeprefix("OFT_H")), operator))
    if not pairs:
        return None
    _, operator = min(pairs)
    return int(label["operator_executed_oft_steps"].get(operator, 0))


def collect_state(
    *,
    pool: Any,
    state_key: str,
    state_meta: Mapping[str, Any],
    label: Mapping[str, Any],
    bundle: Mapping[str, Any],
    client: Any,
    capture: StableLatentCapture,
    boundaries: list[int],
    chunk_steps: int,
    adapter: Mapping[str, Any],
    libero_plus_root: str | None,
) -> dict[str, Any]:
    from rase.collect.forked_rollout import restore_pool_state, rollout_seed
    from rase.collect.oracle_continuation import (
        OracleChunkContinuation,
        raw_libero_to_oracle_arrays,
    )
    from rase.collect.policy_step import as_batched_action, current_timestep
    from rase.collect.pool_candidates import observation_from_libero_env

    restored = restore_pool_state(
        pool,
        state_key,
        libero_plus_root=libero_plus_root,
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    # A second same-task environment is reserved for counterfactual Student
    # steps.  The persistent trajectory is never restored after it starts.
    branch_restored = restore_pool_state(
        pool,
        state_key,
        libero_plus_root=libero_plus_root,
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    rows: list[dict[str, Any]] = []
    try:
        restored.forkable.restore(
            restored.snapshot, check_task_fingerprint=restored.check_task_fingerprint
        )
        single = restored.handle.vector_env.envs[0]
        vector_env = restored.handle.vector_env
        branch_single = branch_restored.handle.vector_env.envs[0]
        branch_vector_env = branch_restored.handle.vector_env
        task = str(
            getattr(single, "task_description", "")
            or restored.loaded.metadata.instruction
        )
        horizon = int(getattr(single, "_max_episode_steps", 600))
        shared_seed = rollout_seed(state_key, 0, 0, salt=2_026_080_403)
        oft = OracleChunkContinuation(client, instruction=task)
        oft.bind_control_env(restored.handle.control_env)
        oft.reset()
        observation = observation_from_libero_env(single)
        elapsed = 0
        terminal = False
        trajectory_success = False
        stop_reason = "horizon"
        boundary_set = set(boundaries)

        # Continue the persistent trajectory after the last recorded boundary
        # so replay parity is checked against the original persistent outcome.
        while not terminal:
            timestep = current_timestep(restored.handle.control_env)
            if timestep >= horizon:
                stop_reason = "horizon"
                break

            if elapsed in boundary_set:
                boundary_snapshot = restored.forkable.snapshot()
                current_latent, student_chunk = query_student(
                    bundle,
                    capture,
                    observation,
                    task=task,
                    seed=shared_seed,
                    chunk_steps=chunk_steps,
                )
                _, _, proprio = raw_libero_to_oracle_arrays(restored.handle.control_env)

                # Student one-step counterfactual runs in an isolated same-task
                # environment.  Preserve process RNGs because ForkableEnv
                # restore includes Python/NumPy global states.
                with preserve_rng_state():
                    branch_restored.forkable.restore(
                        boundary_snapshot,
                        check_task_fingerprint=branch_restored.check_task_fingerprint,
                    )
                    branch_observation = observation_from_libero_env(branch_single)
                    student_obs, _, st_term, st_trunc, st_info = branch_vector_env.step(
                        as_batched_action(student_chunk[0])
                    )
                    st_terminated, st_truncated, st_success = _success_from_step(
                        st_term, st_trunc, st_info
                    )
                    if st_terminated or st_truncated:
                        next_latent_student = np.zeros(
                            capture.latent_dim, dtype=np.float32
                        )
                    else:
                        next_latent_student, _ = query_student(
                            bundle,
                            capture,
                            student_obs,
                            task=task,
                            seed=shared_seed,
                            chunk_steps=1,
                        )
                handback_rollout = rollout_student_handback(
                    branch_restored=branch_restored,
                    boundary_snapshot=boundary_snapshot,
                    bundle=bundle,
                    task=task,
                    seed=shared_seed,
                )

                # Advance the untouched main trajectory with OFT.  The OFT
                # action queue is intentionally persistent across boundaries.
                oft_action = np.asarray(oft.act(observation, task=task), dtype=np.float32)
                oft_obs, _, of_term, of_trunc, of_info = vector_env.step(
                    as_batched_action(oft_action)
                )
                of_terminated, of_truncated, of_success = _success_from_step(
                    of_term, of_trunc, of_info
                )
                if of_terminated or of_truncated:
                    next_latent_oft = np.zeros(capture.latent_dim, dtype=np.float32)
                else:
                    next_latent_oft, _ = query_student(
                        bundle,
                        capture,
                        oft_obs,
                        task=task,
                        seed=shared_seed,
                        chunk_steps=1,
                    )

                operator = _operator_for_boundary(elapsed)
                historical_handback_success = bool(label["operator_success"][operator])
                rows.append(
                    {
                        "schema_version": "rase-pre-c0-r4-boundary-transition/v3",
                        "state_key": state_key,
                        "task_id": str(label["task_id"]),
                        "concrete_task_id": str(label.get("concrete_task_id", state_meta.get("concrete_task_id", "unknown"))),
                        "episode_id": str(label.get("episode_id", state_meta.get("episode_id", "unknown"))),
                        "suite": str(label.get("suite", state_meta.get("suite", "unknown"))),
                        "split": str(label.get("split", state_meta.get("split", "unknown"))),
                        "elapsed_oft_steps": elapsed,
                        "simulator_timestep": timestep,
                        "horizon": horizon,
                        "latent": current_latent.tolist(),
                        "proprio": proprio.tolist(),
                        "student_action_chunk": student_chunk.tolist(),
                        "student_action": student_chunk[0].tolist(),
                        "oft_action": oft_action.tolist(),
                        "next_latent_student": next_latent_student.tolist(),
                        "next_latent_oft": next_latent_oft.tolist(),
                        "student_step_terminal": bool(st_terminated or st_truncated),
                        "student_step_success": st_success,
                        "oft_step_terminal": bool(of_terminated or of_truncated),
                        "oft_step_success": of_success,
                        "success_if_handback_now": bool(handback_rollout["success"]),
                        "handback_continuation_steps": int(handback_rollout["steps"]),
                        "handback_stop_reason": str(handback_rollout["stop_reason"]),
                        "historical_success_if_handback_now": historical_handback_success,
                        "success_if_continue_oft": None,
                        "student_failure_risk": not bool(handback_rollout["success"]),
                        "base_success_at_h0": bool(label["base_success"]),
                        "minimum_successful_oft_steps": None,
                        "persistent_executed_oft_steps": None,
                        "remaining_teacher_steps": None,
                    }
                )
                observation = oft_obs
                terminal = of_terminated or of_truncated
                trajectory_success = of_success if terminal else False
                if terminal:
                    stop_reason = "success" if of_success else (
                        "terminated" if of_terminated else "truncated"
                    )
                elapsed += 1
                continue

            oft_action = np.asarray(oft.act(observation, task=task), dtype=np.float32)
            observation, _, term, trunc, info = vector_env.step(as_batched_action(oft_action))
            terminated, truncated, success = _success_from_step(term, trunc, info)
            terminal = terminated or truncated
            trajectory_success = success if terminal else False
            if terminal:
                stop_reason = "success" if success else ("terminated" if terminated else "truncated")
            elapsed += 1

        successful_boundaries = [
            int(row["elapsed_oft_steps"])
            for row in rows
            if bool(row["success_if_handback_now"])
        ]
        minimum_successful_steps = min(successful_boundaries, default=None)
        for row in rows:
            row["success_if_continue_oft"] = bool(trajectory_success)
            row["persistent_executed_oft_steps"] = int(elapsed)
            row["remaining_teacher_steps"] = max(
                0, int(elapsed) - int(row["elapsed_oft_steps"])
            )
            row["minimum_successful_oft_steps"] = minimum_successful_steps
        historical_persistent = bool(label["operator_success"]["OFT_PERSISTENT"])
        finite_safe_boundaries = [step for step in successful_boundaries if step > 0]
        # The cost-aware oracle is evaluated on every audited state. If no live
        # handback branch succeeds it falls back to the persistent trajectory.
        live_oracle_steps = min(successful_boundaries, default=int(elapsed))
        return {
            "schema_version": "rase-pre-c0-r4-state-transitions/v3",
            "state_key": state_key,
            "task_id": str(label["task_id"]),
            "suite": str(label.get("suite", "unknown")),
            "split": str(label.get("split", "unknown")),
            "n_boundaries": len(rows),
            "persistent_replay_success": trajectory_success,
            "expected_persistent_success": historical_persistent,
            "persistent_replay_match": trajectory_success == historical_persistent,
            "stop_reason": stop_reason,
            "executed_oft_steps": elapsed,
            "minimum_successful_handback_boundary": minimum_successful_steps,
            "finite_safe": bool(finite_safe_boundaries),
            "live_oracle_executed_oft_steps": live_oracle_steps,
            "historical_handback_label_matches": sum(
                bool(row["success_if_handback_now"])
                == bool(row["historical_success_if_handback_now"])
                for row in rows
            ),
            "historical_handback_labels_compared": len(rows),
            "rows": rows,
        }
    finally:
        branch_restored.close()
        restored.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-keys-json", type=Path, required=True)
    parser.add_argument("--opportunity-audit", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--boundary", type=int, action="append", default=[])
    parser.add_argument("--chunk-steps", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=20260808)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--state-key", action="append", default=[],
        help="Optional exact-state filter; may be repeated for targeted parity checks",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    boundaries = args.boundary or [0, 8, 16, 32, 64, 96, 128]
    if boundaries != sorted(set(boundaries)) or boundaries[0] != 0:
        raise ValueError("boundaries must be sorted, unique, and begin at 0")

    cfg = _load(args.config.resolve())
    keys_payload = _load(args.state_keys_json.resolve())
    audit = _load(args.opportunity_audit.resolve())
    labels = _labels_by_state(audit)
    state_meta = {
        str(row["state_key"]): row for row in keys_payload.get("records", [])
    }

    from scripts.generate_oft_pool_candidates import _suite
    from rase.backends.lerobot_libero_plus import _patch_lerobot_init_states
    from rase.backends.libero_plus_paths import ensure_libero_plus_paths
    from rase.collect.forked_rollout import load_smolvla_policy_bundle
    from rase.collect.state_pool import StatePool
    from rase.oracle.client import OracleClient

    adapter = dict(cfg.get("adapter") or {})
    libero_plus_root = _expand(adapter.get("libero_plus_root"), "LIBERO_PLUS_ROOT")
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()
    pool_path = Path(_expand(cfg.get("pool"), "RASE_POOL_ROOT") or "pool")
    if not pool_path.is_absolute():
        pool_path = ROOT / pool_path
    pool = StatePool(pool_path.resolve())

    selected = []
    requested = set(args.state_key)
    for state_key, meta in state_meta.items():
        if requested and state_key not in requested:
            continue
        if state_key not in labels or str(meta.get("split")) != args.split:
            continue
        pool_meta = pool.read_state(state_key, load_observations=False).metadata
        if _suite(pool_meta.task_id) == args.suite:
            selected.append(state_key)
    selected.sort()
    if args.max_states > 0:
        selected = selected[: args.max_states]
    selected = selected[args.shard_index :: args.num_shards]
    missing_requested = sorted(requested - set(selected))
    if missing_requested:
        raise ValueError(
            f"requested states are absent from suite/split selection: {missing_requested}"
        )

    client = OracleClient(args.endpoint, timeout_ms=60_000)
    model_info = client.model_info()
    if model_info.get("suite") not in {None, args.suite}:
        raise ValueError(f"oracle suite mismatch: {model_info.get('suite')} != {args.suite}")

    policy_path = Path(
        _expand(adapter.get("policy_path"), "RASE_POLICY_PATH")
        or "ckpts/smolvla_libero"
    )
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    bundle = load_smolvla_policy_bundle(
        policy_path.resolve(),
        device=str(adapter.get("device", "cuda")),
        num_steps=int(adapter.get("num_steps", 10)),
        n_action_steps=int(adapter.get("n_action_steps", 10)),
        tokenizer_path=_expand(adapter.get("tokenizer_path"), "RASE_TOKENIZER_PATH"),
        observation_height=int(adapter.get("observation_height", 360)),
        observation_width=int(adapter.get("observation_width", 360)),
    )
    capture = StableLatentCapture(
        bundle, latent_dim=args.latent_dim, seed=args.projection_seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = args.output_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for index, state_key in enumerate(selected, start=1):
        target = state_dir / f"{state_key}.json"
        if target.is_file() and not args.force:
            result = json.loads(target.read_text(encoding="utf-8"))
        else:
            result = collect_state(
                pool=pool,
                state_key=state_key,
                state_meta=state_meta[state_key],
                label=labels[state_key],
                bundle=bundle,
                client=client,
                capture=capture,
                boundaries=boundaries,
                chunk_steps=args.chunk_steps,
                adapter=adapter,
                libero_plus_root=libero_plus_root,
            )
            target.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(result)
        print(
            f"R4_BOUNDARY suite={args.suite} state={state_key} "
            f"index={index}/{len(selected)} n={result['n_boundaries']} "
            f"replay_match={result['persistent_replay_match']}",
            flush=True,
        )

    rows = [row for result in outputs for row in result.get("rows", [])]
    jsonl_path = args.output_dir / f"boundaries_{args.suite}.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "rase-pre-c0-r4-boundary-collection/v3",
        "suite": args.suite,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "split": args.split,
        "n_states": len(outputs),
        "n_rows": len(rows),
        "persistent_replay_matches": sum(
            bool(row.get("persistent_replay_match")) for row in outputs
        ),
        "historical_handback_label_matches": sum(
            int(row.get("historical_handback_label_matches", 0)) for row in outputs
        ),
        "historical_handback_labels_compared": sum(
            int(row.get("historical_handback_labels_compared", 0)) for row in outputs
        ),
        "state_summaries": [
            {key: row.get(key) for key in (
                "state_key", "task_id", "suite", "persistent_replay_success",
                "expected_persistent_success", "persistent_replay_match",
                "executed_oft_steps", "minimum_successful_handback_boundary",
                "finite_safe", "live_oracle_executed_oft_steps",
                "historical_handback_label_matches",
                "historical_handback_labels_compared",
            )}
            for row in outputs
        ],
        "boundaries": boundaries,
        "latent_dim": args.latent_dim,
        "projection_seed": args.projection_seed,
        "projection_sha256": capture.projection_sha256,
        "oracle_model_info": dict(model_info),
        "collector_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            args.config.resolve().read_bytes()
        ).hexdigest(),
        "state_keys_sha256": hashlib.sha256(
            args.state_keys_json.resolve().read_bytes()
        ).hexdigest(),
        "opportunity_audit_sha256": hashlib.sha256(
            args.opportunity_audit.resolve().read_bytes()
        ).hexdigest(),
        "counterfactual_env_isolation": "separate_same-task_environment",
        "source_audit": str(args.opportunity_audit.resolve()),
        "source_state_keys": str(args.state_keys_json.resolve()),
        "output": str(jsonl_path.resolve()),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["persistent_replay_matches"] == report["n_states"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
