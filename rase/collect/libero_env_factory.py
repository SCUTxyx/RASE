"""Build a LIBERO-Plus ControlEnv for a pool ``task_id`` without loading a policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rase.backends.lerobot_libero_plus import (
    _patch_lerobot_init_states,
    catalog_task_to_suite_index,
)
from rase.backends.libero_plus_paths import ensure_libero_plus_paths

_TASK_ID_RE = re.compile(
    r"^(?P<suite>libero_(?:spatial|object|goal|10))_(?P<id>\d{6})$"
)


@dataclass(frozen=True)
class ParsedTaskId:
    suite: str
    catalog_task_id: int


@dataclass
class LiberoEnvHandle:
    """Owns a SyncVectorEnv; expose the ControlEnv used by ForkableEnv."""

    vector_env: Any
    control_env: Any
    suite: str
    catalog_task_id: int
    task_index: int

    def close(self) -> None:
        self.vector_env.close()


def parse_pool_task_id(task_id: str) -> ParsedTaskId:
    match = _TASK_ID_RE.fullmatch(str(task_id))
    if match is None:
        raise ValueError(
            f"unsupported pool task_id {task_id!r}; expected "
            "libero_{spatial|object|goal|10}_NNNNNN"
        )
    return ParsedTaskId(
        suite=match.group("suite"),
        catalog_task_id=int(match.group("id")),
    )


def _resolve_plus_task_index(suite: Any, catalog_task_id: int) -> int:
    """Map pool catalog id → suite index; require LIBERO-Plus (not clean-10)."""
    task_index = catalog_task_to_suite_index(catalog_task_id)
    n_tasks = len(suite.tasks)
    if task_index < 0 or task_index >= n_tasks:
        hint = ""
        if n_tasks <= 10 and catalog_task_id > n_tasks:
            hint = (
                " — installed `libero` looks like clean LIBERO (10 tasks/suite). "
                "Reinstall LIBERO-Plus editable: "
                "`pip install -e $LIBERO_PLUS_ROOT` and ensure site-packages "
                "does not shadow it with a stale `libero/` copy."
            )
        raise ValueError(
            f"{suite.name} task_id {catalog_task_id} is out of range "
            f"(n_tasks={n_tasks}){hint}"
        )
    return task_index


def make_libero_env_for_task(
    task_id: str,
    *,
    init_state_id: int,
    seed: int = 0,
    observation_height: int = 360,
    observation_width: int = 360,
    libero_plus_root: str | None = None,
) -> LiberoEnvHandle:
    """Create one in-process LiberoEnv matching collection geometry (no policy)."""
    # Must run before any `libero` import; benchmark init reads BDDL paths.
    ensure_libero_plus_paths(libero_plus_root)
    _patch_lerobot_init_states()

    import gymnasium as gym
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

    parsed = parse_pool_task_id(task_id)
    suite_cls = benchmark.get_benchmark_dict()[parsed.suite]
    suite = suite_cls()
    task_index = _resolve_plus_task_index(suite, parsed.catalog_task_id)
    from lerobot.envs.libero import get_task_init_states

    n_init_states = len(get_task_init_states(suite, task_index))
    if init_state_id < 0 or init_state_id >= n_init_states:
        raise ValueError(
            f"init_state_id {init_state_id} out of range for {n_init_states} init states"
        )

    def make_single() -> LiberoEnv:
        return LiberoEnv(
            task_suite=suite,
            task_id=task_index,
            task_suite_name=parsed.suite,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            init_states=True,
            episode_index=int(init_state_id),
            n_envs=1,
            obs_type="pixels_agent_pos",
            observation_height=observation_height,
            observation_width=observation_width,
            control_mode="relative",
        )

    vector_env = gym.vector.SyncVectorEnv([make_single])
    single = vector_env.envs[0]
    vector_env.reset(seed=[int(seed)])
    return LiberoEnvHandle(
        vector_env=vector_env,
        control_env=single._env,
        suite=parsed.suite,
        catalog_task_id=parsed.catalog_task_id,
        task_index=task_index,
    )
