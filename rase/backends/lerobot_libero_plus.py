"""Task-level LeRobot evaluation backend for classified LIBERO-Plus tasks.

LeRobot 0.5.1 only exposes suite-level LIBERO evaluation. Collapse evaluation
needs per-task selection from the Plus catalog, Plus path resolution, and
Plus-aware init-state loading (`_view_` / `_table_` / ... suffixes).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rase.backends.libero_plus_paths import ensure_libero_plus_paths
from rase.envs.task_catalog import LiberoPlusTask
from rase.eval.collapse import CollapseError, atomic_write_json

_LOCK = threading.Lock()
_PATCHED = False
_POLICY_CACHE: dict[
    tuple[str, str, int, int, str | None, str | None], dict[str, Any]
] = {}


def catalog_task_to_suite_index(task_id: int) -> int:
    """Catalog IDs are 1-based; LeRobot / Plus suite indices are 0-based."""
    if int(task_id) < 1:
        raise CollapseError(f"catalog task_id must be >= 1, got {task_id}")
    return int(task_id) - 1


def _patch_lerobot_init_states() -> None:
    """Replace LeRobot's naive init-state path join with Plus Benchmark logic."""
    global _PATCHED
    if _PATCHED:
        return
    import lerobot.envs.libero as lerobot_libero
    import torch

    def get_task_init_states(task_suite: Any, i: int):
        # Plus Benchmark.get_task_init_states uses torch.load without
        # weights_only=False; PyTorch >=2.6 defaults break numpy pickles.
        if not hasattr(task_suite, "get_task_init_states"):
            raise CollapseError(
                "task suite lacks get_task_init_states; is LIBERO-Plus installed?"
            )
        original_load = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = _load  # type: ignore[assignment]
        try:
            return task_suite.get_task_init_states(i)
        finally:
            torch.load = original_load  # type: ignore[assignment]

    lerobot_libero.get_task_init_states = get_task_init_states
    _PATCHED = True


def _resolve_suite_task_index(suite: Any, task: LiberoPlusTask) -> int:
    index = catalog_task_to_suite_index(task.task_id)
    n_tasks = len(suite.tasks)
    if index >= n_tasks:
        raise CollapseError(
            f"{task.suite} task_id {task.task_id} is out of range "
            f"(suite has {n_tasks} tasks)"
        )
    suite_name = suite.tasks[index].name
    if suite_name == task.name:
        return index
    for candidate, record in enumerate(suite.tasks):
        if record.name == task.name:
            return candidate
    raise CollapseError(
        f"suite/task mismatch for {task.key}: catalog name={task.name!r}, "
        f"suite[{index}]={suite_name!r}"
    )


def _evaluation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = dict(config.get("evaluation") or {})
    episodes = int(evaluation.get("episodes_per_task") or 1)
    if episodes < 1:
        raise CollapseError("episodes_per_task must be a positive integer")
    batch_size = int(evaluation.get("batch_size") or 1)
    if batch_size < 1:
        raise CollapseError("batch_size must be a positive integer")
    return {
        "episodes_per_task": episodes,
        "batch_size": min(batch_size, episodes),
        "seed": int(evaluation.get("seed") or 0),
        "num_steps": int(evaluation.get("num_steps") or 10),
        "n_action_steps": int(evaluation.get("n_action_steps") or 10),
        "device": str(evaluation.get("device") or "cuda"),
        "max_episodes_rendered": int(evaluation.get("max_episodes_rendered") or 0),
    }


def _resolve_local_vlm_path(tokenizer_path: Path | None) -> str | None:
    """Return an absolute local VLM/tokenizer dir, or None if unset.

    SmolVLA's config defaults ``vlm_model_name`` to a HuggingFace hub id. In
    offline / mirror-flaky environments that triggers a network fetch even when
    ``ckpts/SmolVLM2-500M-Instruct`` is already on disk. Pointing both the VLM
    backbone and the preprocessor tokenizer at the local directory avoids hub
    access.
    """
    if tokenizer_path is None:
        return None
    resolved = Path(tokenizer_path).expanduser().resolve()
    if not resolved.is_dir():
        raise CollapseError(f"tokenizer/VLM path does not exist: {resolved}")
    if not (
        (resolved / "config.json").is_file()
        or (resolved / "tokenizer_config.json").is_file()
    ):
        raise CollapseError(
            "tokenizer/VLM path lacks config.json or tokenizer_config.json "
            f"(not a HF model/tokenizer dir): {resolved}"
        )
    return str(resolved)


def _get_or_load_policy(
    policy_path: Path,
    *,
    device: str,
    num_steps: int,
    n_action_steps: int,
    env_cfg: Any,
    tokenizer_path: Path | None = None,
    action_tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    resolved_tokenizer = _resolve_local_vlm_path(tokenizer_path)
    resolved_action_tokenizer = _resolve_local_vlm_path(action_tokenizer_path)
    key = (
        str(policy_path.resolve()), device, num_steps, n_action_steps,
        resolved_tokenizer, resolved_action_tokenizer,
    )
    with _LOCK:
        cached = _POLICY_CACHE.get(key)
        if cached is not None:
            return cached

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path))
        policy_cfg.pretrained_path = str(policy_path)
        policy_cfg.device = device
        if hasattr(policy_cfg, "n_action_steps"):
            policy_cfg.n_action_steps = n_action_steps
        if hasattr(policy_cfg, "num_steps"):
            policy_cfg.num_steps = num_steps
        # Evaluation must not inherit training-oriented memory/compile knobs from
        # public checkpoints.  In particular, Pi0/Pi0Fast checkpoints may ship
        # with these enabled even though no gradients are computed here.
        if hasattr(policy_cfg, "compile_model"):
            policy_cfg.compile_model = False
        if hasattr(policy_cfg, "gradient_checkpointing"):
            policy_cfg.gradient_checkpointing = False
        # Must rewrite before make_policy: VLM weights load from vlm_model_name.
        if resolved_tokenizer is not None and hasattr(policy_cfg, "vlm_model_name"):
            policy_cfg.vlm_model_name = resolved_tokenizer
        if resolved_tokenizer is not None and hasattr(policy_cfg, "text_tokenizer_name"):
            policy_cfg.text_tokenizer_name = resolved_tokenizer
        if (
            resolved_action_tokenizer is not None
            and hasattr(policy_cfg, "action_tokenizer_name")
        ):
            policy_cfg.action_tokenizer_name = resolved_action_tokenizer

        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        policy.eval()

        preprocessor_overrides = {
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        }
        if resolved_tokenizer is not None:
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": resolved_tokenizer
            }
        # Only FAST checkpoints contain the separate action-tokenizer processor.
        # Pi0.5 uses ``tokenizer_processor`` plus its own state-tokenizer step;
        # injecting a FAST-only override makes LeRobot reject the saved pipeline.
        if resolved_action_tokenizer is not None:
            action_overrides = {}
            action_overrides["action_tokenizer_name"] = resolved_action_tokenizer
            # LeRobot's FAST processor constructs a second PaliGemma tokenizer
            # internally; overriding only ``tokenizer_processor`` is insufficient.
            if resolved_tokenizer is not None:
                action_overrides["paligemma_tokenizer_name"] = resolved_tokenizer
            preprocessor_overrides["action_tokenizer_processor"] = action_overrides
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(policy_path),
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy_cfg
        )
        cached = {
            "policy": policy,
            "policy_cfg": policy_cfg,
            "preprocessor": preprocessor,
            "postprocessor": postprocessor,
            "env_preprocessor": env_preprocessor,
            "env_postprocessor": env_postprocessor,
        }
        _POLICY_CACHE[key] = cached
        return cached


def evaluate(
    task: LiberoPlusTask,
    task_output_dir: Path,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one classified Plus task and return JSON-serializable metrics."""
    ensure_libero_plus_paths()
    _patch_lerobot_init_states()

    settings = _evaluation_settings(resolved_config)
    policy_path_value = resolved_config.get("policy_path")
    if not policy_path_value:
        raise CollapseError("resolved_config.policy_path is required for evaluation")
    policy_path = Path(str(policy_path_value)).expanduser()
    if not policy_path.exists():
        raise CollapseError(f"policy path does not exist: {policy_path}")

    task_output_dir = Path(task_output_dir)
    task_output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env
    from lerobot.envs.utils import close_envs
    from lerobot.scripts.lerobot_eval import eval_policy
    from libero.libero import benchmark

    suite_index = None
    try:
        suite_cls = benchmark.get_benchmark_dict()[task.suite]
    except KeyError as exc:
        raise CollapseError(f"unknown LIBERO suite {task.suite!r}") from exc
    suite = suite_cls()
    suite_index = _resolve_suite_task_index(suite, task)

    env_cfg = LiberoEnvConfig(
        task=task.suite,
        task_ids=[suite_index],
        obs_type="pixels_agent_pos",
        init_states=True,
    )
    bundle = _get_or_load_policy(
        policy_path,
        device=settings["device"],
        num_steps=settings["num_steps"],
        n_action_steps=settings["n_action_steps"],
        env_cfg=env_cfg,
        tokenizer_path=(
            Path(str(resolved_config["tokenizer_path"]))
            if resolved_config.get("tokenizer_path")
            else None
        ),
        action_tokenizer_path=(
            Path(str(resolved_config["action_tokenizer_path"]))
            if resolved_config.get("action_tokenizer_path")
            else None
        ),
    )

    envs = make_env(env_cfg, n_envs=settings["batch_size"], use_async_envs=False)
    try:
        vec_env = envs[task.suite][suite_index]
        videos_dir = task_output_dir / "videos"
        max_rendered = settings["max_episodes_rendered"]
        with torch.no_grad():
            result = eval_policy(
                env=vec_env,
                policy=bundle["policy"],
                env_preprocessor=bundle["env_preprocessor"],
                env_postprocessor=bundle["env_postprocessor"],
                preprocessor=bundle["preprocessor"],
                postprocessor=bundle["postprocessor"],
                n_episodes=settings["episodes_per_task"],
                max_episodes_rendered=max_rendered,
                videos_dir=videos_dir if max_rendered > 0 else None,
                start_seed=settings["seed"],
            )
    finally:
        close_envs(envs)

    aggregated = dict(result["aggregated"])
    metrics = {
        "pc_success": float(aggregated.get("pc_success", 0.0)),
        "avg_sum_reward": float(aggregated.get("avg_sum_reward", 0.0)),
        "avg_max_reward": float(aggregated.get("avg_max_reward", 0.0)),
        "eval_s": float(aggregated.get("eval_s", 0.0)),
        "eval_ep_s": float(aggregated.get("eval_ep_s", 0.0)),
        "episodes": settings["episodes_per_task"],
        "suite_task_index": suite_index,
        "catalog_task_id": int(task.task_id),
        "task_name": task.name,
        "n_action_steps": settings["n_action_steps"],
        "num_steps": settings["num_steps"],
        "seed": settings["seed"],
        "per_episode": list(result.get("per_episode") or []),
        "video_paths": list(result.get("video_paths") or []),
    }
    atomic_write_json(task_output_dir / "metrics.json", metrics)
    # Keep a compact human-readable summary beside the full metrics.
    summary = {
        "task_key": task.key,
        "pc_success": metrics["pc_success"],
        "episodes": metrics["episodes"],
        "suite_task_index": suite_index,
    }
    (task_output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics
