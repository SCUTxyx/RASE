"""Real SmolVLA + LIBERO-Plus adapter for NGC Step-1 state collection."""

from __future__ import annotations

import io
import json
import math
import os
import time
from collections import deque
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rase.backends.lerobot_libero_plus import (
    _get_or_load_policy,
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
from rase.collect.perturb_sampler import PerturbationRequest
from rase.collect.pipeline import EpisodeResult, EpisodeSnapshot
from rase.envs.forkable_env import ForkableEnv
from rase.interventions.decision_context import (
    build_decision_context,
    finalize_source_suffix_parity,
    strict_continue_suffix,
)

_CLEAN_CONTROL_PROTOCOLS = frozenset(
    {"W9B-clean-control/v1", "W9C-clean-control/v1"}
)

_SUITE_NAMES = {
    "Spatial": "libero_spatial",
    "Object": "libero_object",
    "Goal": "libero_goal",
    "Long": "libero_10",
}
_DIRECT_CATEGORIES = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "layout": "Objects Layout",
}
_OTHER_CATEGORIES = {
    "light": "Light Conditions",
    "background": "Background Textures",
    "noise": "Sensor Noise",
}


@dataclass(frozen=True)
class _CatalogTask:
    suite: str
    task_id: int
    name: str
    category: str
    level: int


def _expand_path(value: object | None, env_name: str) -> Path:
    raw = str(value) if value else os.environ.get(env_name)
    if not raw:
        raise ValueError(f"path required via adapter config or {env_name}")
    return Path(os.path.expandvars(raw)).expanduser().resolve()


def _category_for(request: PerturbationRequest) -> str:
    if request.dimension == "other":
        try:
            return _OTHER_CATEGORIES[request.subdimension]
        except KeyError as exc:
            raise ValueError(
                f"unsupported other subdimension {request.subdimension!r}"
            ) from exc
    if request.dimension == "combination":
        raise ValueError(
            "LIBERO-Plus task_classification.json has no camera+robot combination "
            "category. Use the camera/robot pilot config; combination synthesis "
            "requires a separately validated paired-task protocol."
        )
    try:
        return _DIRECT_CATEGORIES[request.dimension]
    except KeyError as exc:
        raise ValueError(f"unsupported perturbation dimension {request.dimension!r}") from exc


def _load_catalog(path: Path) -> dict[str, tuple[_CatalogTask, ...]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read LIBERO-Plus task catalog {path}: {exc}") from exc
    result: dict[str, tuple[_CatalogTask, ...]] = {}
    for suite, records in raw.items():
        tasks: list[_CatalogTask] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"{suite} contains a non-object task")
            # Upstream marks 121 Light Conditions tasks with null difficulty.
            level_raw = record.get("difficulty_level")
            if level_raw is None:
                continue
            try:
                level = int(level_raw)
                task_id = int(record["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid catalog record in {suite}: {record}") from exc
            if level not in range(1, 6):
                continue
            tasks.append(
                _CatalogTask(
                    suite=suite,
                    task_id=task_id,
                    name=str(record["name"]),
                    category=str(record["category"]),
                    level=level,
                )
            )
        result[suite] = tuple(tasks)
    return result


def select_catalog_task(
    catalog: Mapping[str, tuple[_CatalogTask, ...]],
    request: PerturbationRequest,
) -> _CatalogTask:
    try:
        suite = _SUITE_NAMES[request.suite]
    except KeyError as exc:
        raise ValueError(f"unknown suite quota label {request.suite!r}") from exc
    if request.dimension == "clean":
        # Official clean-10 names — NOT Plus suite indices 0–9 (those are
        # `_table_` / `_view_` layout variants in the expanded Plus catalog).
        task_id = (
            int(request.task_id)
            if request.task_id is not None
            else int(request.seed % N_CLEAN_TASKS) + 1
        )
        if task_id not in range(1, N_CLEAN_TASKS + 1):
            raise ValueError(
                f"clean task_id must be in [1, {N_CLEAN_TASKS}], got {task_id}"
            )
        name = clean_task_name(suite, task_id)
        assert_clean_task_name(name)
        return _CatalogTask(
            suite=suite,
            task_id=task_id,
            name=name,
            category="Clean Control",
            level=0,
        )
    category = _category_for(request)
    candidates = tuple(
        task
        for task in catalog.get(suite, ())
        if task.category == category and task.level == request.level
    )
    if not candidates:
        raise ValueError(
            f"no LIBERO-Plus task for suite={suite}, category={category}, "
            f"level=L{request.level}"
        )
    return candidates[request.seed % len(candidates)]


def _png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HWC uint8 RGB image, got {array.shape}/{array.dtype}")
    stream = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def _observation_images(observation: Mapping[str, Any]) -> dict[str, bytes]:
    pixels = observation.get("pixels")
    if not isinstance(pixels, Mapping):
        raise ValueError("LIBERO observation is missing pixels")
    names = {"image": "agentview", "image2": "wrist"}
    return {
        names.get(str(name), str(name).lower()): _png_bytes(np.asarray(value)[0])
        for name, value in pixels.items()
    }


def _queued_env_action_suffix(bundle: Mapping[str, Any], policy: Any) -> np.ndarray:
    """Copy SmolVLA's not-yet-executed queue into LIBERO env action space."""
    from lerobot.utils.constants import ACTION

    queues = getattr(policy, "_queues", None)
    queue = queues.get(ACTION) if isinstance(queues, Mapping) else None
    if queue is None or not queue:
        raise ValueError("source policy has no active action suffix at snapshot time")
    actions = []
    for queued in list(queue):
        action = bundle["postprocessor"](queued.detach().clone())
        transition = bundle["env_postprocessor"]({ACTION: action})
        array = np.asarray(transition[ACTION].detach().cpu().numpy(), dtype=np.float32)
        if array.shape == (1, 7):
            array = array[0]
        if array.shape != (7,):
            raise ValueError(f"queued action has invalid env-space shape {array.shape}")
        actions.append(array)
    return np.stack(actions, axis=0)


def _snapshot(
    forkable: ForkableEnv,
    observation: Mapping[str, Any],
    proprio: np.ndarray,
    chunk_index: int,
    *,
    env_step: int | None = None,
    source_policy: str | None = None,
    action_chunk_size: int | None = None,
    action_chunk_offset: int | None = None,
    active_action_suffix: np.ndarray | None = None,
    public_history: tuple[tuple[int, Mapping[str, bytes], np.ndarray], ...] = (),
    public_action_history: tuple[np.ndarray, ...] = (),
) -> EpisodeSnapshot:
    snapshot = forkable.snapshot()
    payload = dict(snapshot.payload)
    images = _observation_images(observation)
    controller_state = {
        "snapshot_format": "rase.forkable_env/v1",
        "task_fingerprint": snapshot.task_fingerprint,
        "env_counters": payload["env_counters"],
        "robots": payload["robots"],
        "observables": payload["observables"],
        "obs_cache": payload["obs_cache"],
    }
    if active_action_suffix is not None:
        if None in (env_step, source_policy, action_chunk_size, action_chunk_offset):
            raise ValueError("decision-context v2 requires complete chunk provenance")
        history_refs = []
        proprio_history = []
        for history_index, (history_step, history_images, history_proprio) in enumerate(
            public_history
        ):
            refs = {}
            for camera, image in history_images.items():
                key = f"hist_{history_index:02d}_{camera}"
                images[key] = image
                refs[camera] = key
            history_refs.append({"env_step": int(history_step), "observations": refs})
            proprio_history.append(np.asarray(history_proprio, dtype=np.float32))
        controller_state["decision_context"] = build_decision_context(
            source_policy=str(source_policy),
            snapshot_env_step=int(env_step),
            action_chunk_size=int(action_chunk_size),
            action_chunk_offset=int(action_chunk_offset),
            active_action_suffix=active_action_suffix,
            public_action_history=public_action_history,
            public_proprio_history=proprio_history,
            public_observation_history=history_refs,
        )
    return EpisodeSnapshot(
        step=chunk_index,
        sim_state=np.asarray(payload["sim_state"]).copy(),
        controller_state=controller_state,
        rng_state=payload["rng"],
        observations=images,
        proprio=np.asarray(proprio, dtype=np.float32).copy(),
    )


def _supports_strict_continue(snapshot: EpisodeSnapshot) -> bool:
    try:
        strict_continue_suffix(snapshot.controller_state)
    except ValueError:
        return False
    return True


def _success_from_info(info: Mapping[str, Any]) -> bool:
    from rase.collect.policy_step import success_from_info

    return success_from_info(info)


def _validated_init_state_id(
    request: PerturbationRequest, *, n_init_states: int, required: bool
) -> int:
    if request.init_state_id is None:
        if required:
            raise ValueError("scheduled collection requires init_state_id")
        # Legacy protocols did not carry init provenance. Avoid a fixed production
        # episode_index while preserving deterministic behavior.
        init_state_id = int(request.index) % n_init_states
    else:
        init_state_id = int(request.init_state_id)
    if init_state_id < 0 or init_state_id >= n_init_states:
        raise ValueError(
            f"init_state_id {init_state_id} out of range for {n_init_states} init states"
        )
    return init_state_id


def _lerobot_env_kwargs(
    *,
    suite: Any,
    task_index: int,
    suite_name: str,
    camera_name: Any,
    init_state_id: int,
    obs_type: str,
    observation_height: int,
    observation_width: int,
    control_mode: str,
) -> dict[str, Any]:
    """Single source of truth for explicit schedule→LeRobot episode mapping."""
    return {
        "task_suite": suite,
        "task_id": task_index,
        "task_suite_name": suite_name,
        "camera_name": camera_name,
        "init_states": True,
        "episode_index": init_state_id,
        "n_envs": 1,
        "obs_type": obs_type,
        "observation_height": observation_height,
        "observation_width": observation_width,
        "control_mode": control_mode,
    }


class LeRobotLiberoPlusCollectionAdapter:
    """Runs one deterministic episode and snapshots every N action chunks.

    Clean controls use the official 10-task LIBERO suites (by frozen name).
    Failure / challenge dimensions still use LIBERO-Plus catalog selection.
    """

    def __init__(self, config: Mapping[str, Any]):
        adapter = dict(config.get("adapter_config") or {})
        collection = dict(config["collection"])
        protocol_version = dict(config.get("protocol") or {}).get("version")
        self.require_init_state_id = protocol_version in _CLEAN_CONTROL_PROTOCOLS
        self.libero_plus_root = adapter.get("libero_plus_root")
        self.libero_clean_root = adapter.get("libero_clean_root")
        ensure_libero_plus_paths(self.libero_plus_root)
        _patch_lerobot_init_states()

        paths = ensure_libero_plus_paths(self.libero_plus_root)
        catalog_path = Path(paths["benchmark_root"]) / "benchmark" / "task_classification.json"
        self.catalog = _load_catalog(catalog_path)
        self.policy_path = _expand_path(adapter.get("policy_path"), "RASE_POLICY_PATH")
        if not self.policy_path.is_dir():
            raise ValueError(f"policy path does not exist: {self.policy_path}")
        tokenizer = adapter.get("tokenizer_path") or os.environ.get("RASE_TOKENIZER_PATH")
        self.tokenizer_path = (
            Path(os.path.expandvars(str(tokenizer))).expanduser().resolve()
            if tokenizer
            else None
        )
        if self.tokenizer_path is not None and not self.tokenizer_path.is_dir():
            raise ValueError(f"tokenizer path does not exist: {self.tokenizer_path}")
        self.device = str(adapter.get("device", "cuda"))
        self.num_steps = int(adapter.get("num_steps", 10))
        self.n_action_steps = int(adapter.get("n_action_steps", 10))
        self.capture_decision_context_v2 = bool(
            adapter.get("capture_decision_context_v2", False)
        )
        self.snapshot_offset_steps = int(adapter.get("snapshot_offset_steps", 0))
        self.public_history_steps = int(adapter.get("public_history_steps", 4))
        if self.capture_decision_context_v2:
            if not 0 < self.snapshot_offset_steps < self.n_action_steps:
                raise ValueError(
                    "decision-context v2 requires snapshot_offset_steps strictly "
                    "inside the action chunk"
                )
            if self.public_history_steps < 1:
                raise ValueError("public_history_steps must be positive")
        self.observation_height = int(adapter.get("observation_height", 360))
        self.observation_width = int(adapter.get("observation_width", 360))
        max_chunks = collection.get("max_action_chunks")
        self.max_action_chunks = int(max_chunks) if max_chunks is not None else None

    def _resolve_suite_and_task(
        self, request: PerturbationRequest
    ) -> tuple[Any, _CatalogTask, int, str]:
        selected = select_catalog_task(self.catalog, request)
        if request.dimension == "clean":
            ensure_libero_clean_paths(self.libero_clean_root)
            suite = build_clean_suite(
                selected.suite, clean_root=self.libero_clean_root
            )
            if suite.n_tasks != N_CLEAN_TASKS:
                raise ValueError(
                    f"clean suite {selected.suite} has n_tasks={suite.n_tasks}, "
                    f"expected {N_CLEAN_TASKS}"
                )
            task_index = int(selected.task_id) - 1
            live_name = str(suite.tasks[task_index].name)
            assert_clean_task_name(live_name)
            if live_name != selected.name:
                raise ValueError(
                    f"clean catalog/suite mismatch at {selected.suite}:"
                    f"{selected.task_id}: catalog={selected.name!r} live={live_name!r}"
                )
            return suite, selected, task_index, "clean"

        ensure_libero_plus_paths(self.libero_plus_root)
        from libero.libero import benchmark

        suite_cls = benchmark.get_benchmark_dict()[selected.suite]
        suite = suite_cls()
        task_index = catalog_task_to_suite_index(selected.task_id)
        if selected.name and suite.tasks[task_index].name != selected.name:
            raise ValueError(
                f"catalog/suite mismatch at {selected.suite}:{selected.task_id}"
            )
        return suite, selected, task_index, "plus"

    def run_episode(
        self, request: PerturbationRequest, episode_id: str, cadence: int
    ) -> EpisodeResult:
        import gymnasium as gym
        import torch
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.libero import LiberoEnv, get_task_init_states
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION, OBS_STATE
        from lerobot.utils.random_utils import set_seed

        suite, selected, task_index, libero_flavor = self._resolve_suite_and_task(
            request
        )
        init_state_id = _validated_init_state_id(
            request,
            n_init_states=len(get_task_init_states(suite, task_index)),
            required=self.require_init_state_id,
        )

        env_cfg = LiberoEnvConfig(
            task=selected.suite,
            task_ids=[task_index],
            obs_type="pixels_agent_pos",
            init_states=True,
            observation_height=self.observation_height,
            observation_width=self.observation_width,
        )
        bundle = _get_or_load_policy(
            self.policy_path,
            device=self.device,
            num_steps=self.num_steps,
            n_action_steps=self.n_action_steps,
            env_cfg=env_cfg,
            tokenizer_path=self.tokenizer_path,
        )
        # Keep one environment in-process: ForkableEnv captures process-global NumPy RNG.
        def make_single() -> LiberoEnv:
            return LiberoEnv(
                **_lerobot_env_kwargs(
                    suite=suite,
                    task_index=task_index,
                    suite_name=selected.suite,
                    camera_name=env_cfg.camera_name,
                    init_state_id=init_state_id,
                    obs_type=env_cfg.obs_type,
                    observation_height=self.observation_height,
                    observation_width=self.observation_width,
                    control_mode=env_cfg.control_mode,
                )
            )

        vector_env = gym.vector.SyncVectorEnv([make_single])
        single = vector_env.envs[0]
        policy = bundle["policy"]
        set_seed(request.seed)
        episode_started = time.perf_counter()
        policy.reset()
        observation, _ = vector_env.reset(seed=[request.seed])
        forkable = ForkableEnv(single._env)

        max_steps = int(single._max_episode_steps)
        if self.max_action_chunks is not None:
            max_steps = min(max_steps, self.max_action_chunks * self.n_action_steps)
        max_chunks = math.ceil(max_steps / self.n_action_steps)
        snapshots: list[EpisodeSnapshot] = []
        public_history: deque[tuple[int, Mapping[str, bytes], np.ndarray]] = deque(
            maxlen=self.public_history_steps
        )
        public_action_history: deque[np.ndarray] = deque(
            maxlen=self.public_history_steps
        )
        pending_suffix_checks: list[dict[str, Any]] = []
        success = False
        env_steps = 0
        policy_select_calls = 0
        policy_select_elapsed_s = 0.0
        device_type = next(policy.parameters()).device.type
        amp_context = (
            torch.autocast(device_type=device_type)
            if bool(getattr(policy.config, "use_amp", False))
            else nullcontext()
        )
        try:
            with torch.no_grad(), amp_context:
                for env_step in range(max_steps):
                    policy_observation = preprocess_observation(observation)
                    policy_observation["task"] = [single.task_description]
                    env_observation = bundle["env_preprocessor"](policy_observation)
                    state = env_observation.get(OBS_STATE)
                    if state is None:
                        raise ValueError(
                            "LIBERO processor did not produce observation.state"
                        )
                    if self.capture_decision_context_v2:
                        public_history.append(
                            (
                                env_step,
                                _observation_images(observation),
                                state.detach().cpu().numpy()[0].copy(),
                            )
                        )
                    snapshot_offset = (
                        self.snapshot_offset_steps
                        if self.capture_decision_context_v2
                        else 0
                    )
                    if env_step % self.n_action_steps == snapshot_offset:
                        chunk_index = env_step // self.n_action_steps
                        if chunk_index % cadence == 0:
                            suffix = (
                                _queued_env_action_suffix(bundle, policy)
                                if self.capture_decision_context_v2
                                else None
                            )
                            episode_snapshot = _snapshot(
                                forkable,
                                observation,
                                state.detach().cpu().numpy()[0],
                                chunk_index,
                                env_step=env_step,
                                source_policy=f"smolvla:{self.policy_path.name}",
                                action_chunk_size=self.n_action_steps,
                                action_chunk_offset=snapshot_offset,
                                active_action_suffix=suffix,
                                public_history=tuple(public_history),
                                public_action_history=tuple(public_action_history),
                            )
                            snapshots.append(episode_snapshot)
                            if suffix is not None:
                                pending_suffix_checks.append(
                                    {
                                        "snapshot": episode_snapshot,
                                        "expected": suffix,
                                        "observed": [],
                                    }
                                )

                    processed = bundle["preprocessor"](env_observation)
                    select_started = time.perf_counter()
                    action = policy.select_action(processed)
                    policy_select_elapsed_s += time.perf_counter() - select_started
                    policy_select_calls += 1
                    action = bundle["postprocessor"](action)
                    transition = bundle["env_postprocessor"]({ACTION: action})
                    action_numpy = transition[ACTION].detach().cpu().numpy()
                    if self.capture_decision_context_v2:
                        current_action = np.asarray(
                            action_numpy, dtype=np.float32
                        ).reshape(7).copy()
                        for check in tuple(pending_suffix_checks):
                            observed = check["observed"]
                            if len(observed) < len(check["expected"]):
                                observed.append(current_action)
                            if len(observed) == len(check["expected"]):
                                context = check["snapshot"].controller_state[
                                    "decision_context"
                                ]
                                finalize_source_suffix_parity(
                                    context, np.stack(observed, axis=0)
                                )
                                pending_suffix_checks.remove(check)
                    observation, _, terminated, truncated, info = vector_env.step(
                        action_numpy
                    )
                    env_steps = env_step + 1
                    if self.capture_decision_context_v2:
                        public_action_history.append(
                            np.asarray(action_numpy, dtype=np.float32).reshape(7).copy()
                        )
                    if bool(terminated[0]) or bool(truncated[0]):
                        success = _success_from_info(info)
                        break
        finally:
            vector_env.close()

        # A cap is a deliberate failure unless success happened before it.
        outcome = "success" if success else "failure"
        if self.capture_decision_context_v2:
            snapshots = [
                snapshot
                for snapshot in snapshots
                if _supports_strict_continue(snapshot)
            ]
        if not snapshots and max_chunks > 0:
            raise RuntimeError("episode produced no snapshots")
        bddl_stem = selected.name if libero_flavor == "clean" else None
        if bddl_stem:
            assert_clean_task_name(bddl_stem)
        return EpisodeResult(
            outcome=outcome,
            task_id=f"{selected.suite}_{selected.task_id:06d}",
            instruction=str(single.task_description),
            snapshots=tuple(snapshots),
            suite=request.suite,
            init_state_id=init_state_id,
            clean_task_name=bddl_stem,
            bddl_stem=bddl_stem,
            libero_flavor=libero_flavor,
            metrics={
                "env_steps": env_steps,
                "episode_wall_s": round(time.perf_counter() - episode_started, 6),
                "policy_select_calls": policy_select_calls,
                "policy_select_elapsed_s": round(policy_select_elapsed_s, 6),
                "policy_ms_per_env_step": (
                    round(1000 * policy_select_elapsed_s / env_steps, 6)
                    if env_steps
                    else None
                ),
            },
        )


def make_adapter(config: Mapping[str, Any]) -> LeRobotLiberoPlusCollectionAdapter:
    return LeRobotLiberoPlusCollectionAdapter(config)
