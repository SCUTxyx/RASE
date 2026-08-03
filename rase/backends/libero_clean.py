"""Official clean LIBERO (10 tasks/suite) without using Plus suite indices 0–9.

Plus expands each suite to thousands of `_table_` / `_view_` variants. Catalog
IDs 1–10 therefore do **not** mean the original clean tasks. This module builds
an in-process 10-task suite from the frozen official names and points
``get_libero_path`` at assets that contain the exact-name BDDL/init files.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from rase.eval.collapse import CollapseError

CLEAN_TASK_NAMES_SCHEMA = "rase-clean-libero-task-names/v1"
N_CLEAN_TASKS = 10
# Plus layout/viewpoint/etc. suffixes are trailing tokens like `_table_12`,
# not substrings inside official names (e.g. `from_table_center`).
_PERTURB_SUFFIX_RE = re.compile(
    r"(_table_\d+|_view_\d+|_tb_\d+|_light_\d+|_noise_\d+|"
    r"_robot_\d+|_background_\d+|_language_\d+|_add_\d+|_level\d*)"
    r"($|\.)"
)

_SUITE_ALIASES = {
    "Spatial": "libero_spatial",
    "Object": "libero_object",
    "Goal": "libero_goal",
    "Long": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_10": "libero_10",
}


def default_clean_task_names_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "clean_libero_task_names.json"


def load_clean_task_names(
    path: str | Path | None = None,
) -> dict[str, tuple[str, ...]]:
    catalog_path = Path(path) if path else default_clean_task_names_path()
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollapseError(f"cannot read clean task catalog {catalog_path}: {exc}") from exc
    if payload.get("schema_version") != CLEAN_TASK_NAMES_SCHEMA:
        raise CollapseError(
            f"unsupported clean task catalog schema in {catalog_path}"
        )
    suites = payload.get("suites")
    if not isinstance(suites, Mapping):
        raise CollapseError("clean task catalog missing suites object")
    result: dict[str, tuple[str, ...]] = {}
    for suite, names in suites.items():
        if not isinstance(names, list) or len(names) != N_CLEAN_TASKS:
            raise CollapseError(
                f"{suite} must list exactly {N_CLEAN_TASKS} clean task names"
            )
        cleaned: list[str] = []
        for name in names:
            text = str(name)
            assert_clean_task_name(text)
            cleaned.append(text)
        result[str(suite)] = tuple(cleaned)
    for required in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        if required not in result:
            raise CollapseError(f"clean task catalog missing suite {required}")
    return result


def normalize_suite_name(suite: str) -> str:
    try:
        return _SUITE_ALIASES[suite]
    except KeyError as exc:
        raise CollapseError(f"unknown suite label {suite!r}") from exc


def assert_clean_task_name(name: str) -> None:
    if not name or not isinstance(name, str):
        raise CollapseError("clean task name must be a non-empty string")
    if _PERTURB_SUFFIX_RE.search(name):
        raise CollapseError(
            f"refusing perturbed/Plus-variant task name as clean control: {name!r}"
        )


def clean_task_name(suite: str, clean_task_id: int, *, catalog: Mapping[str, tuple[str, ...]] | None = None) -> str:
    """Map 1-based clean_task_id to the frozen official task name."""
    names = (catalog or load_clean_task_names())[normalize_suite_name(suite)]
    if int(clean_task_id) not in range(1, N_CLEAN_TASKS + 1):
        raise CollapseError(
            f"clean_task_id must be in [1, {N_CLEAN_TASKS}], got {clean_task_id}"
        )
    name = names[int(clean_task_id) - 1]
    assert_clean_task_name(name)
    return name


def resolve_libero_clean_root(explicit: str | Path | None = None) -> Path:
    """Resolve a checkout that contains exact-name clean BDDL/init assets."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    env_root = os.environ.get("LIBERO_ROOT") or os.environ.get("LIBERO_CLEAN_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    # Prefer vanilla LIBERO; Plus disk also stores unsuffixed BDDL/init files.
    candidates.extend(
        (
            Path("/root/autodl-tmp/src/LIBERO"),
            Path(os.environ["LIBERO_PLUS_ROOT"])
            if os.environ.get("LIBERO_PLUS_ROOT")
            else None,
            Path("/root/autodl-tmp/src/LIBERO-plus"),
        )
    )
    for candidate in candidates:
        if candidate is None:
            continue
        root = _as_benchmark_root(candidate)
        if root is not None and _has_clean_assets(root):
            return candidate.resolve()
    raise CollapseError(
        "clean LIBERO assets not found. Set LIBERO_ROOT to a checkout containing "
        "exact-name BDDL/init files for the official 10-task suites."
    )


def _as_benchmark_root(path: Path) -> Path | None:
    path = path.expanduser()
    if (path / "bddl_files").is_dir() and (path / "init_files").is_dir():
        return path.resolve()
    nested = path / "libero" / "libero"
    if (nested / "bddl_files").is_dir() and (nested / "init_files").is_dir():
        return nested.resolve()
    return None


def _has_clean_assets(root: Path) -> bool:
    catalog = load_clean_task_names()
    for suite, names in catalog.items():
        for name in names:
            bddl = root / "bddl_files" / suite / f"{name}.bddl"
            init = root / "init_files" / suite / f"{name}.pruned_init"
            if not bddl.is_file() or not init.is_file():
                return False
    return True


def build_libero_clean_path_dict(clean_root: str | Path | None = None) -> dict[str, str]:
    resolved = resolve_libero_clean_root(clean_root)
    root = _as_benchmark_root(resolved)
    if root is None:
        raise CollapseError("resolved clean LIBERO path is missing bddl_files/init_files")
    datasets = root.parent / "datasets"
    return {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(datasets),
        "assets": str(root / "assets"),
    }


def default_libero_clean_config_dir() -> Path:
    override = os.environ.get("LIBERO_CLEAN_CONFIG_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".libero_clean_rase").resolve()


def ensure_libero_clean_paths(clean_root: str | Path | None = None) -> dict[str, str]:
    """Point ``get_libero_path`` at clean assets without mutating ``~/.libero``."""
    paths = build_libero_clean_path_dict(clean_root)
    for key in ("bddl_files", "init_states", "assets"):
        if not Path(paths[key]).is_dir():
            raise CollapseError(f"clean LIBERO {key} directory missing: {paths[key]}")

    config_dir = default_libero_clean_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    existing: Mapping[str, object] | None = None
    if config_file.is_file():
        try:
            loaded = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, Mapping):
                existing = loaded
        except (OSError, yaml.YAMLError):
            existing = None
    if existing != paths:
        config_file.write_text(
            yaml.safe_dump(paths, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )

    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    try:
        import libero.libero as libero_pkg

        libero_pkg.libero_config_path = str(config_dir)
        libero_pkg.config_file = str(config_file)
    except Exception:
        pass
    return paths


class CleanLiberoSuite:
    """Minimal 10-task suite compatible with LeRobot ``LiberoEnv``."""

    def __init__(self, suite_name: str, tasks: tuple[Any, ...]):
        if len(tasks) != N_CLEAN_TASKS:
            raise CollapseError(
                f"CleanLiberoSuite requires {N_CLEAN_TASKS} tasks, got {len(tasks)}"
            )
        self.name = suite_name
        self.tasks = list(tasks)
        self.n_tasks = N_CLEAN_TASKS

    def get_task(self, i: int) -> Any:
        return self.tasks[i]

    def get_num_tasks(self) -> int:
        return self.n_tasks

    def get_task_names(self) -> list[str]:
        return [task.name for task in self.tasks]

    def get_task_init_states(self, i: int) -> Any:
        """Load exact-name init states (Plus Benchmark skips unsuffixed paths)."""
        import torch
        from libero.libero import get_libero_path

        task = self.tasks[i]
        assert_clean_task_name(task.name)
        init_states_path = (
            Path(get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file
        )
        if not init_states_path.is_file():
            raise CollapseError(f"missing clean init states: {init_states_path}")
        return torch.load(init_states_path, weights_only=False)


def build_clean_suite(
    suite: str,
    *,
    clean_root: str | Path | None = None,
    catalog: Mapping[str, tuple[str, ...]] | None = None,
) -> CleanLiberoSuite:
    """Build a clean 10-task suite and ensure path config points at clean assets."""
    from libero.libero.benchmark import Task, grab_language_from_filename

    ensure_libero_clean_paths(clean_root)
    suite_name = normalize_suite_name(suite)
    names = (catalog or load_clean_task_names())[suite_name]
    root = Path(build_libero_clean_path_dict(clean_root)["benchmark_root"])
    tasks: list[Any] = []
    for name in names:
        assert_clean_task_name(name)
        bddl = root / "bddl_files" / suite_name / f"{name}.bddl"
        init = root / "init_files" / suite_name / f"{name}.pruned_init"
        if not bddl.is_file():
            raise CollapseError(f"missing clean BDDL: {bddl}")
        if not init.is_file():
            raise CollapseError(f"missing clean init states: {init}")
        # Match LeRobot/LIBERO: pass ``name.bddl`` so SCENE* prefixes are stripped
        # and the trailing ``.bddl`` is removed (bare ``replace('_',' ')`` breaks Long).
        language = grab_language_from_filename(suite_name, f"{name}.bddl")
        if not language or "SCENE" in language:
            raise CollapseError(
                f"invalid clean task language for {name!r}: {language!r}"
            )
        tasks.append(
            Task(
                name=name,
                language=language,
                problem="Libero",
                problem_folder=suite_name,
                bddl_file=f"{name}.bddl",
                init_states_file=f"{name}.pruned_init",
            )
        )
    suite_obj = CleanLiberoSuite(suite_name, tuple(tasks))
    if suite_obj.n_tasks != N_CLEAN_TASKS:
        raise CollapseError("clean suite n_tasks invariant violated")
    return suite_obj


def resolve_clean_task_index(
    suite: str,
    clean_task_id: int,
    *,
    catalog: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[int, str]:
    name = clean_task_name(suite, clean_task_id, catalog=catalog)
    return int(clean_task_id) - 1, name
