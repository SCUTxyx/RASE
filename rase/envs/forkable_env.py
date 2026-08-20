"""Strict state-fork wrapper for LIBERO-plus / robosuite 1.4 environments."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .snapshot import EnvSnapshot, SnapshotError


class UnsupportedEnvironmentError(RuntimeError):
    """Raised instead of silently producing an incomplete snapshot."""


class TaskMismatchError(SnapshotError):
    """Raised when restoring a snapshot into a different task or model."""


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedEnvironmentError("ForkableEnv requires NumPy") from exc
    return np


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedEnvironmentError(
            "ForkableEnv full-state restore requires the mujoco bindings"
        ) from exc
    return mujoco


def _qualified_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class CompatibilityProfile:
    """Explicit class allowlists for one tested simulator stack."""

    wrapper_classes: frozenset[str]
    task_module_prefixes: tuple[str, ...]
    controller_classes: frozenset[str]
    robot_classes: frozenset[str]
    linear_interpolator_classes: frozenset[str]
    delta_buffer_classes: frozenset[str]
    ring_buffer_classes: frozenset[str]


LIBERO_ROBOSUITE_140 = CompatibilityProfile(
    wrapper_classes=frozenset(
        {
            "libero.libero.envs.env_wrapper.ControlEnv",
            "libero.libero.envs.env_wrapper.OffScreenRenderEnv",
            "libero.libero.envs.env_wrapper.SegmentationRenderEnv",
        }
    ),
    task_module_prefixes=("libero.libero.envs.",),
    controller_classes=frozenset({"robosuite.controllers.osc.OperationalSpaceController"}),
    robot_classes=frozenset({"robosuite.robots.single_arm.SingleArm"}),
    linear_interpolator_classes=frozenset(
        {"robosuite.controllers.interpolators.linear_interpolator.LinearInterpolator"}
    ),
    delta_buffer_classes=frozenset({"robosuite.utils.buffers.DeltaBuffer"}),
    ring_buffer_classes=frozenset({"robosuite.utils.buffers.RingBuffer"}),
)


_ENV_COUNTER_FIELDS = ("cur_time", "timestep", "done")
_CONTROLLER_FIELDS = (
    "goal_ori",
    "goal_pos",
    "relative_ori",
    "ori_ref",
    "kp",
    "kd",
    "new_update",
    "torques",
    "initial_joint",
    "initial_ee_pos",
    "initial_ee_ori_mat",
    "action_scale",
    "action_input_transform",
    "action_output_transform",
    # Runtime caches written by ``update()`` and read by ``run_controller()``.
    # When ``new_update`` is False (e.g. at the moment a boundary snapshot is
    # taken mid-episode), ``run_controller()`` consumes these cached values as-is;
    # leaving them out of the snapshot let a restored env replay the *previous*
    # run's stale cache and diverge from the live trajectory.
    "ee_pos",
    "ee_ori_mat",
    "ee_pos_vel",
    "ee_ori_vel",
    "joint_pos",
    "joint_vel",
    "J_pos",
    "J_ori",
    "J_full",
    "mass_matrix",
)
# Controller fields present in historical StatePool bundles (built before the
# runtime-cache fields above were captured).  Legacy pool payloads have no
# ``mujoco_data`` and no cache fields, so their controller restore must use
# this narrower, byte-compatible set.
_CONTROLLER_CACHE_FIELDS = frozenset(
    {
        "ee_pos",
        "ee_ori_mat",
        "ee_pos_vel",
        "ee_ori_vel",
        "joint_pos",
        "joint_vel",
        "J_pos",
        "J_ori",
        "J_full",
        "mass_matrix",
    }
)
_CONTROLLER_LEGACY_FIELDS = tuple(
    field for field in _CONTROLLER_FIELDS if field not in _CONTROLLER_CACHE_FIELDS
)
_INTERPOLATOR_FIELDS = ("step", "start", "goal")
_ROBOT_VALUE_FIELDS = ("torques",)
_ROBOT_DELTA_BUFFERS = (
    "recent_qpos",
    "recent_actions",
    "recent_torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_acc",
)
_ROBOT_RING_BUFFERS = ("recent_ee_vel_buffer",)
_OBSERVABLE_FIELDS = (
    "_time_since_last_sample",
    "_current_delay",
    "_current_observed_value",
    "_sampled",
)
_RNG_ATTRS = ("np_random", "_np_random")

# These describe the compiled MuJoCo state-space / object topology and do not
# change while an episode is running.  In contrast, ``model.get_xml()`` is not
# a stable identity source in mujoco-py: lazy renderer setup and task runtime
# code can update values that are serialized back into the XML.
_MODEL_TOPOLOGY_FIELDS = (
    "nq",
    "nv",
    "nu",
    "na",
    "nbody",
    "njnt",
    "ngeom",
    "nsite",
    "ncam",
    "nlight",
    "nmesh",
    "ntex",
    "nmat",
    "neq",
    "ntendon",
    "nsensor",
    "nkey",
)


def _require_attributes(obj: Any, fields: Iterable[str], role: str) -> None:
    missing = [field for field in fields if not hasattr(obj, field)]
    if missing:
        raise UnsupportedEnvironmentError(
            f"{role} {_qualified_name(obj)} is missing required state fields: {missing}"
        )


def _copy_fields(obj: Any, fields: Iterable[str]) -> dict[str, Any]:
    return {field: copy.deepcopy(getattr(obj, field)) for field in fields}


def _restore_fields(obj: Any, state: Mapping[str, Any], fields: Iterable[str], role: str) -> None:
    expected = set(fields)
    if set(state) != expected:
        raise SnapshotError(
            f"{role} state keys differ: expected {sorted(expected)}, got {sorted(state)}"
        )
    _require_attributes(obj, fields, role)
    for field in fields:
        setattr(obj, field, copy.deepcopy(state[field]))


# `bvh_active` is the broad-phase bounding-volume-hierarchy activity buffer,
# a ~1.1 MB internal acceleration structure that mj_forward() rebuilds
# deterministically on every step.  Storing it would dominate snapshot size for
# no benefit.  Everything else in raw MjData -- including warm-start / solver
# buffers that mj_forward() does *not* recompute (qacc_warmstart, ...) and
# state like qpos/qvel/act/time/ctrl/userdata/mocap -- is captured verbatim.
_MUJOCO_DATA_EXCLUDE = frozenset({"bvh_active"})


def _raw_mujoco_data(sim: Any) -> Any:
    data = getattr(sim, "data", None)
    raw = getattr(data, "_data", None)
    if raw is None:
        raise UnsupportedEnvironmentError(
            "sim.data must expose a raw mujoco _data for full-state restore"
        )
    return raw


def _raw_mujoco_model(sim: Any) -> Any:
    data = getattr(sim, "data", None)
    model = getattr(data, "_model", None)
    raw_model = getattr(model, "_model", None)
    if raw_model is None:
        raise UnsupportedEnvironmentError(
            "sim.data._model must expose a raw mujoco model for full-state restore"
        )
    return raw_model


def _capture_mujoco_data(sim: Any) -> dict[str, Any]:
    """Capture every raw MjData field that can affect a future sim.step().

    A restored ``mjData`` must be bit-identical to the live one before the next
    step for the fork to be side-effect free.  ``sim.get_state()`` only covers
    qpos/qvel/act/udd_state; the solver warm-start buffers (qacc_warmstart,
    dof_island, efc_JT, ...) and counters (ncon, nefc, solver_iter) are left
    stale by set_state + forward and silently change collision / solving.
    """
    np = _numpy()
    raw = _raw_mujoco_data(sim)
    fields: dict[str, Any] = {}
    for name in dir(raw):
        if name.startswith("_") or name in _MUJOCO_DATA_EXCLUDE:
            continue
        try:
            value = getattr(raw, name)
        except Exception:
            continue
        if isinstance(value, np.ndarray):
            fields[name] = np.ascontiguousarray(value).copy()
        elif isinstance(value, (bool, int, float)):
            fields[name] = value
    return fields


def _restore_mujoco_data(sim: Any, fields: Mapping[str, Any]) -> None:
    """Swap in a fresh raw MjData populated verbatim from a captured snapshot.

    Creating a fresh MjData and assigning the captured fields resets every
    internal buffer that forward()/step() are allowed to mutate, so the
    read-only solver buffers (efc_JT, efc_aref, ...) start from their canonical
    state instead of whatever the live data last computed.  Fields that cannot
    be assigned (sized buffers written during the previous solve) are left at
    their zeroed/identity defaults.

    ``sim.forward()`` is deliberately *not* called here.  The captured data is
    the exact post-step state of the live env, and a fresh ``mj_forward(qpos)``
    recomputes FK geometry differently from the solver-corrected post-step
    values (contact-settled free bodies keep their corrected ``xpos``/``geom_*``
    across steps; a fresh forward snaps them back).  Any read between restore
    and the next step -- e.g. the boundary feature extraction -- must see the
    same geometry the live source rollout rendered.  The next ``env.step()``
    calls ``sim.forward()`` itself, so stepping stays correct.
    """
    np = _numpy()
    mujoco = _mujoco()
    model = _raw_mujoco_model(sim)
    fresh = mujoco.MjData(model)
    for name, value in fields.items():
        try:
            if isinstance(value, np.ndarray):
                setattr(fresh, name, np.ascontiguousarray(value).copy())
            else:
                setattr(fresh, name, value)
        except Exception:
            # Read-only or size-mismatched buffer: leave the fresh default in
            # place; the next mj_forward()/mj_step() rebuilds these.
            continue
    # Swap the underlying raw MjData of the live wrapper (robosuite MjData
    # delegates attribute access to ``._data``), so every external reference
    # to ``sim.data`` still sees the restored state.
    sim.data._data = fresh


def _capture_rng_object(rng: Any) -> dict[str, Any]:
    np = _numpy()
    if isinstance(rng, np.random.Generator):
        return {
            "kind": "generator",
            "bit_generator": type(rng.bit_generator).__name__,
            "state": copy.deepcopy(rng.bit_generator.state),
        }
    if isinstance(rng, np.random.RandomState):
        return {"kind": "random_state", "state": copy.deepcopy(rng.get_state())}
    raise UnsupportedEnvironmentError(
        f"unsupported environment RNG class {_qualified_name(rng)}"
    )


def _restore_rng_object(rng: Any, state: Mapping[str, Any]) -> None:
    np = _numpy()
    if state.get("kind") == "generator" and isinstance(rng, np.random.Generator):
        if type(rng.bit_generator).__name__ != state.get("bit_generator"):
            raise SnapshotError("environment RNG bit-generator differs from snapshot")
        rng.bit_generator.state = copy.deepcopy(state["state"])
    elif state.get("kind") == "random_state" and isinstance(rng, np.random.RandomState):
        rng.set_state(copy.deepcopy(state["state"]))
    else:
        raise SnapshotError("environment RNG kind differs from snapshot")


class ForkableEnv:
    """Wrap one in-process LIBERO environment with deterministic snapshot/restore.

    The default profile intentionally supports only the inspected LIBERO-plus
    and robosuite 1.4 state layout. Pass a separate explicit profile for test
    doubles or after auditing a new upstream release.
    """

    def __init__(
        self,
        env: Any,
        *,
        compatibility: CompatibilityProfile = LIBERO_ROBOSUITE_140,
    ) -> None:
        self.env = env
        self.compatibility = compatibility
        self._validate_stack()
        self._task_fingerprint = self._compute_task_fingerprint()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def task_fingerprint(self) -> str:
        return self._task_fingerprint

    def step(self, action: Any) -> Any:
        return self.env.step(action)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self.env.reset(*args, **kwargs)
        # A hard reset may replace the underlying MuJoCo model and controllers.
        self._validate_stack()
        self._task_fingerprint = self._compute_task_fingerprint()
        return result

    @property
    def _task_env(self) -> Any:
        return self.env.env

    def _validate_stack(self) -> None:
        profile = self.compatibility
        wrapper_name = _qualified_name(self.env)
        if wrapper_name not in profile.wrapper_classes:
            raise UnsupportedEnvironmentError(
                f"environment wrapper {wrapper_name} is not explicitly allowed"
            )
        if not hasattr(self.env, "env") or not hasattr(self.env, "sim"):
            raise UnsupportedEnvironmentError("wrapper must expose env and sim")

        task_name = _qualified_name(self._task_env)
        if not any(task_name.startswith(prefix) for prefix in profile.task_module_prefixes):
            raise UnsupportedEnvironmentError(
                f"task environment {task_name} is outside allowed module prefixes"
            )
        _require_attributes(self._task_env, _ENV_COUNTER_FIELDS, "task environment")
        _require_attributes(self._task_env, ("robots", "_observables", "_obs_cache"), "task environment")
        if not self._task_env.robots:
            raise UnsupportedEnvironmentError("at least one robot is required")

        for index, robot in enumerate(self._task_env.robots):
            if _qualified_name(robot) not in profile.robot_classes:
                raise UnsupportedEnvironmentError(
                    f"robot[{index}] {_qualified_name(robot)} is not explicitly allowed"
                )
            _require_attributes(
                robot,
                ("controller", *_ROBOT_VALUE_FIELDS, *_ROBOT_DELTA_BUFFERS, *_ROBOT_RING_BUFFERS),
                f"robot[{index}]",
            )
            controller = robot.controller
            if _qualified_name(controller) not in profile.controller_classes:
                raise UnsupportedEnvironmentError(
                    f"controller[{index}] {_qualified_name(controller)} is not explicitly allowed"
                )
            _require_attributes(controller, _CONTROLLER_FIELDS, f"controller[{index}]")
            for name in ("interpolator_pos", "interpolator_ori"):
                if not hasattr(controller, name):
                    raise UnsupportedEnvironmentError(f"controller[{index}] lacks {name}")
                interpolator = getattr(controller, name)
                if (
                    interpolator is not None
                    and _qualified_name(interpolator) not in profile.linear_interpolator_classes
                ):
                    raise UnsupportedEnvironmentError(
                        f"{name} {_qualified_name(interpolator)} is not explicitly allowed"
                    )

    def _compute_task_fingerprint(self) -> str:
        components: dict[str, Any] = {
            "fingerprint_schema": "rase-task-identity/v2",
            "wrapper_class": _qualified_name(self.env),
            "task_class": _qualified_name(self._task_env),
        }
        for source in (self.env, self._task_env):
            for field in (
                "problem_name",
                "domain_name",
                "language_instruction",
                "task_id",
                "bddl_file_name",
            ):
                if field in components:
                    continue
                # Avoid hasattr(): some LIBERO properties raise (e.g. missing
                # parsed_problem["language"]) instead of returning AttributeError.
                try:
                    value = getattr(source, field)
                except Exception:
                    continue
                if value is None:
                    continue
                components[field] = str(value)

        bddl_name = getattr(self._task_env, "bddl_file_name", None)
        if bddl_name:
            path = Path(bddl_name)
            if not path.is_file():
                raise UnsupportedEnvironmentError(f"task BDDL file does not exist: {path}")
            components["bddl_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

        # Prefer an adapter-supplied immutable identity.  Real LIBERO uses a
        # mujoco-py model, for which we fingerprint stable topology rather than
        # get_xml(): that XML can change after renderer initialization or an
        # environment step even though the task/model is still the same.
        if hasattr(self.env.sim, "model_fingerprint"):
            components["model_fingerprint"] = str(self.env.sim.model_fingerprint)
        else:
            model = getattr(self.env.sim, "model", None)
            if model is None:
                raise UnsupportedEnvironmentError(
                    "sim must expose model topology or an explicit model_fingerprint"
                )
            topology: dict[str, Any] = {"class": _qualified_name(model)}
            for field in _MODEL_TOPOLOGY_FIELDS:
                if hasattr(model, field):
                    topology[field] = int(getattr(model, field))
            names = getattr(model, "names", None)
            if names is not None:
                if isinstance(names, str):
                    names_bytes = names.encode("utf-8")
                else:
                    try:
                        names_bytes = bytes(names)
                    except (TypeError, ValueError):
                        names_bytes = repr(names).encode("utf-8")
                topology["names_sha256"] = hashlib.sha256(names_bytes).hexdigest()
            if len(topology) == 1:
                raise UnsupportedEnvironmentError(
                    "sim.model exposes no supported stable topology fields; "
                    "provide sim.model_fingerprint"
                )
            components["model_topology"] = topology

        canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def snapshot(self) -> EnvSnapshot:
        """Capture simulation, counters, controller/robot caches, and RNGs."""

        np = _numpy()
        self._validate_stack()
        fingerprint = self._compute_task_fingerprint()
        if fingerprint != self._task_fingerprint:
            raise UnsupportedEnvironmentError(
                "task/model identity changed since wrapper construction; call reset() first"
            )

        robots = []
        for robot in self._task_env.robots:
            controller = robot.controller
            interpolators = {}
            for name in ("interpolator_pos", "interpolator_ori"):
                interpolator = getattr(controller, name)
                interpolators[name] = (
                    None
                    if interpolator is None
                    else _copy_fields(interpolator, _INTERPOLATOR_FIELDS)
                )
            delta_buffers = {}
            for name in _ROBOT_DELTA_BUFFERS:
                buffer = getattr(robot, name)
                if _qualified_name(buffer) not in self.compatibility.delta_buffer_classes:
                    raise UnsupportedEnvironmentError(
                        f"{name} buffer {_qualified_name(buffer)} is not explicitly allowed"
                    )
                _require_attributes(buffer, ("last", "current"), name)
                delta_buffers[name] = _copy_fields(buffer, ("last", "current"))
            ring_buffers = {}
            for name in _ROBOT_RING_BUFFERS:
                buffer = getattr(robot, name)
                if _qualified_name(buffer) not in self.compatibility.ring_buffer_classes:
                    raise UnsupportedEnvironmentError(
                        f"{name} buffer {_qualified_name(buffer)} is not explicitly allowed"
                    )
                _require_attributes(buffer, ("buf", "ptr", "_size"), name)
                ring_buffers[name] = _copy_fields(buffer, ("buf", "ptr", "_size"))
            robots.append(
                {
                    "class": _qualified_name(robot),
                    "controller_class": _qualified_name(controller),
                    "controller": _copy_fields(controller, _CONTROLLER_FIELDS),
                    "interpolators": interpolators,
                    "values": _copy_fields(robot, _ROBOT_VALUE_FIELDS),
                    "delta_buffers": delta_buffers,
                    "ring_buffers": ring_buffers,
                }
            )

        observables = {}
        for name, observable in sorted(self._task_env._observables.items()):
            _require_attributes(observable, _OBSERVABLE_FIELDS, f"observable {name!r}")
            observables[name] = _copy_fields(observable, _OBSERVABLE_FIELDS)

        env_rngs = {}
        for owner_name, owner in (("wrapper", self.env), ("task", self._task_env)):
            for attr in _RNG_ATTRS:
                if hasattr(owner, attr):
                    env_rngs[f"{owner_name}.{attr}"] = _capture_rng_object(getattr(owner, attr))

        payload = {
            "sim_state": np.asarray(self.env.sim.get_state().flatten()).copy(),
            "mujoco_data": _capture_mujoco_data(self.env.sim),
            "env_counters": _copy_fields(self._task_env, _ENV_COUNTER_FIELDS),
            "robots": robots,
            "observables": observables,
            "obs_cache": copy.deepcopy(self._task_env._obs_cache),
            "rng": {
                "python": copy.deepcopy(random.getstate()),
                "numpy_global": copy.deepcopy(np.random.get_state()),
                "environment": env_rngs,
            },
        }
        return EnvSnapshot(task_fingerprint=fingerprint, payload=payload)

    def restore(
        self, snapshot: EnvSnapshot, *, check_task_fingerprint: bool = True
    ) -> None:
        """Restore in a fixed order and reject cross-task snapshots.

        ``check_task_fingerprint=True`` (default) enforces the stable task
        identity hash: task metadata, BDDL content, and compiled model topology.
        It deliberately excludes ``model.get_xml()``, whose serialized runtime
        values are not invariant within a mujoco-py episode. Legacy v1 pool
        snapshots may still require ``check_task_fingerprint=False`` after an
        external task_id / BDDL bind check and double-restore determinism gate.
        """

        np = _numpy()
        if not isinstance(snapshot, EnvSnapshot):
            raise TypeError("restore expects an EnvSnapshot")
        self._validate_stack()
        current_fingerprint = self._compute_task_fingerprint()
        if (
            check_task_fingerprint
            and snapshot.task_fingerprint != current_fingerprint
        ):
            raise TaskMismatchError(
                "snapshot task fingerprint does not match the current task/model"
            )

        legacy_payload = {
            "sim_state",
            "env_counters",
            "robots",
            "observables",
            "obs_cache",
            "rng",
        }
        full_payload = {*legacy_payload, "mujoco_data"}
        if set(snapshot.payload) not in (legacy_payload, full_payload):
            raise SnapshotError("snapshot payload keys differ from the supported schema")
        state = snapshot.payload

        # 1. MuJoCo state and derived kinematics.  A restored raw MjData must be
        # bit-identical to the live one: set_state_from_flattened + forward
        # leaves solver warm-start / contact buffers stale and silently changes
        # later steps, so we swap in a fresh MjData populated verbatim from the
        # capture (without re-forwarding -- the captured geometry is already the
        # solver-corrected post-step state the live source rollout rendered).
        # Legacy pool bundles (no ``mujoco_data``) keep the old restore path.
        if "mujoco_data" in state:
            _restore_mujoco_data(self.env.sim, state["mujoco_data"])
        else:
            self.env.sim.set_state_from_flattened(np.asarray(state["sim_state"]).copy())
            self.env.sim.forward()

        # 2. Episode bookkeeping.
        _restore_fields(
            self._task_env, state["env_counters"], _ENV_COUNTER_FIELDS, "task environment"
        )

        # 3. Controller goals/interpolators, then robot history buffers.
        if len(state["robots"]) != len(self._task_env.robots):
            raise SnapshotError("robot count differs from snapshot")
        # Legacy pool bundles predate the runtime controller caches; restoring
        # them with the full field set would reject their historical payload.
        controller_fields = _CONTROLLER_FIELDS if "mujoco_data" in state else _CONTROLLER_LEGACY_FIELDS
        for index, (robot, robot_state) in enumerate(
            zip(self._task_env.robots, state["robots"])
        ):
            if robot_state.get("class") != _qualified_name(robot):
                raise SnapshotError(f"robot[{index}] class differs from snapshot")
            controller = robot.controller
            if robot_state.get("controller_class") != _qualified_name(controller):
                raise SnapshotError(f"controller[{index}] class differs from snapshot")
            _restore_fields(
                controller,
                robot_state["controller"],
                controller_fields,
                f"controller[{index}]",
            )
            for name in ("interpolator_pos", "interpolator_ori"):
                interpolator = getattr(controller, name)
                interpolator_state = robot_state["interpolators"][name]
                if (interpolator is None) != (interpolator_state is None):
                    raise SnapshotError(f"controller[{index}].{name} presence differs")
                if interpolator is not None:
                    _restore_fields(
                        interpolator,
                        interpolator_state,
                        _INTERPOLATOR_FIELDS,
                        f"controller[{index}].{name}",
                    )
            _restore_fields(
                robot, robot_state["values"], _ROBOT_VALUE_FIELDS, f"robot[{index}]"
            )
            for name in _ROBOT_DELTA_BUFFERS:
                _restore_fields(
                    getattr(robot, name),
                    robot_state["delta_buffers"][name],
                    ("last", "current"),
                    f"robot[{index}].{name}",
                )
            for name in _ROBOT_RING_BUFFERS:
                _restore_fields(
                    getattr(robot, name),
                    robot_state["ring_buffers"][name],
                    ("buf", "ptr", "_size"),
                    f"robot[{index}].{name}",
                )

        # 4. Task-derived visual state and exact observable scheduling/cache.
        if hasattr(self._task_env, "_post_process"):
            self._task_env._post_process()
        if set(state["observables"]) != set(self._task_env._observables):
            raise SnapshotError("observable names differ from snapshot")
        for name, observable in self._task_env._observables.items():
            _restore_fields(
                observable,
                state["observables"][name],
                _OBSERVABLE_FIELDS,
                f"observable {name!r}",
            )
        self._task_env._obs_cache = copy.deepcopy(state["obs_cache"])

        # 5. RNGs last, rewinding any incidental random draws during restoration.
        rng_state = state["rng"]
        if set(rng_state) != {"python", "numpy_global", "environment"}:
            raise SnapshotError("RNG state keys differ from the supported schema")
        environment_rngs = {}
        for owner_name, owner in (("wrapper", self.env), ("task", self._task_env)):
            for attr in _RNG_ATTRS:
                if hasattr(owner, attr):
                    environment_rngs[f"{owner_name}.{attr}"] = getattr(owner, attr)
        if set(environment_rngs) != set(rng_state["environment"]):
            raise SnapshotError("environment RNG attributes differ from snapshot")
        for key, rng in environment_rngs.items():
            _restore_rng_object(rng, rng_state["environment"][key])
        random.setstate(copy.deepcopy(rng_state["python"]))
        np.random.set_state(copy.deepcopy(rng_state["numpy_global"]))

    def save_snapshot(self, path: str | Path) -> tuple[Path, Path]:
        return self.snapshot().save(path)

    def restore_from(self, path: str | Path) -> None:
        self.restore(EnvSnapshot.load(path))
