"""Build a LIBERO ControlEnv for a pool ``task_id`` without loading a policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from rase.backends.lerobot_libero_plus import (
    _patch_lerobot_init_states,
    catalog_task_to_suite_index,
)
from rase.backends.libero_clean import (
    N_CLEAN_TASKS,
    assert_clean_task_name,
    build_clean_suite,
    clean_task_name,
    ensure_libero_clean_paths,
)
from rase.backends.libero_plus_paths import ensure_libero_plus_paths

_TASK_ID_RE = re.compile(
    r"^(?P<suite>libero_(?:spatial|object|goal|10))_(?P<id>\d{6})$"
)

LiberoFlavor = Literal["clean", "plus"]


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
    libero_flavor: str = "plus"
    clean_task_name: str | None = None

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


def _resolve_clean_task_index(suite: Any, catalog_task_id: int) -> tuple[int, str]:
    if suite.n_tasks != N_CLEAN_TASKS or len(suite.tasks) != N_CLEAN_TASKS:
        raise ValueError(
            f"clean suite must have exactly {N_CLEAN_TASKS} tasks, "
            f"got n_tasks={getattr(suite, 'n_tasks', None)} "
            f"len(tasks)={len(suite.tasks)}"
        )
    if catalog_task_id not in range(1, N_CLEAN_TASKS + 1):
        raise ValueError(
            f"clean catalog_task_id must be in [1, {N_CLEAN_TASKS}], "
            f"got {catalog_task_id}"
        )
    task_index = catalog_task_id - 1
    name = str(suite.tasks[task_index].name)
    assert_clean_task_name(name)
    expected = clean_task_name(suite.name, catalog_task_id)
    if name != expected:
        raise ValueError(
            f"clean task identity mismatch: suite[{task_index}]={name!r}, "
            f"catalog={expected!r}"
        )
    return task_index, name


def make_libero_env_for_task(
    task_id: str,
    *,
    init_state_id: int,
    seed: int = 0,
    observation_height: int = 360,
    observation_width: int = 360,
    libero_plus_root: str | None = None,
    libero_clean_root: str | None = None,
    libero_flavor: LiberoFlavor = "plus",
) -> LiberoEnvHandle:
    """Create one in-process LiberoEnv matching collection geometry (no policy)."""
    parsed = parse_pool_task_id(task_id)
    flavor = str(libero_flavor)
    if flavor not in {"clean", "plus"}:
        raise ValueError(f"libero_flavor must be 'clean' or 'plus', got {flavor!r}")

    if flavor == "clean":
        ensure_libero_clean_paths(libero_clean_root)
        # Plus-patched loader delegates to suite.get_task_init_states; clean suite
        # implements the exact-name path that Plus Benchmark comments out.
        _patch_lerobot_init_states()
        suite = build_clean_suite(parsed.suite, clean_root=libero_clean_root)
        task_index, task_name = _resolve_clean_task_index(suite, parsed.catalog_task_id)
    else:
        ensure_libero_plus_paths(libero_plus_root)
        _patch_lerobot_init_states()
        from libero.libero import benchmark

        suite_cls = benchmark.get_benchmark_dict()[parsed.suite]
        suite = suite_cls()
        task_index = _resolve_plus_task_index(suite, parsed.catalog_task_id)
        task_name = str(suite.tasks[task_index].name)

    import gymnasium as gym
    from lerobot.envs.libero import LiberoEnv, get_task_init_states

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
        libero_flavor=flavor,
        clean_task_name=task_name if flavor == "clean" else None,
    )
