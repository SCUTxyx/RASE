#!/usr/bin/env python3
"""Plan or run a resumable LIBERO-Plus camera/robot collapse evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rase.backends.libero_plus_paths import ensure_libero_plus_paths
from rase.envs.task_catalog import LiberoPlusTaskCatalog, parse_levels
from rase.eval.collapse import (
    CollapseError,
    ResultManifest,
    collect_provenance,
    require_lerobot_backend,
    run_tasks,
)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise CollapseError(f"unresolved environment variable in {value!r}")
        return expanded
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CollapseError(f"cannot load config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CollapseError("config root must be a mapping")
    return _expand_env(value)


def _path_from_args_or_env(
    explicit: str | None, env_name: str, *, child: str | None = None
) -> Path | None:
    value = explicit or os.environ.get(env_name)
    if value:
        return Path(value).expanduser()
    if child and os.environ.get("LIBERO_PLUS_ROOT"):
        return Path(os.environ["LIBERO_PLUS_ROOT"]).expanduser() / child
    return None


def _load_hook(spec: str):
    if ":" not in spec:
        raise CollapseError("backend hook must be formatted module:function")
    module_name, function_name = spec.split(":", 1)
    try:
        hook = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise CollapseError(f"cannot load backend hook {spec}: {exc}") from exc
    if not callable(hook):
        raise CollapseError(f"backend hook {spec} is not callable")
    return hook


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "collapse_camera_robot.yaml"),
    )
    parser.add_argument("--catalog", help="task_classification.json path")
    parser.add_argument("--output-dir")
    parser.add_argument("--env-lock")
    parser.add_argument("--policy-path")
    parser.add_argument("--profile", choices=("smoke", "full"))
    parser.add_argument("--dimensions", help="comma-separated camera,robot")
    parser.add_argument("--levels", help="e.g. L1-L5 or L3,L5")
    parser.add_argument("--suites", help="comma-separated suite names")
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", choices=("lerobot",), default="lerobot")
    parser.add_argument("--backend-hook", help="Python module:function backend")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Skip a task after this many starts (covers native crash retries).",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path, Path | None]:
    raw = _load_yaml(Path(args.config))
    profile_name = args.profile or raw.get("default_profile", "smoke")
    try:
        profile = dict(raw["profiles"][profile_name])
    except (KeyError, TypeError) as exc:
        raise CollapseError(f"profile {profile_name!r} is missing from config") from exc
    selection = dict(raw.get("selection", {}))
    evaluation = dict(raw.get("evaluation", {}))

    dimensions = (
        args.dimensions.split(",") if args.dimensions else selection.get("dimensions")
    )
    levels = parse_levels(args.levels) if args.levels else selection.get("levels")
    suites = args.suites.split(",") if args.suites else selection.get("suites")
    episodes = args.episodes_per_task or profile.get("episodes_per_task")
    if not isinstance(episodes, int) or episodes < 1:
        raise CollapseError("episodes_per_task must be a positive integer")

    catalog = _path_from_args_or_env(
        args.catalog,
        "LIBERO_PLUS_TASK_CATALOG",
        child="libero/libero/benchmark/task_classification.json",
    )
    output = _path_from_args_or_env(args.output_dir, "RASE_COLLAPSE_OUTPUT")
    env_lock = _path_from_args_or_env(args.env_lock, "RASE_ENV_LOCK")
    if env_lock is None and (REPO_ROOT / "env.lock.md").is_file():
        env_lock = REPO_ROOT / "env.lock.md"
    policy = args.policy_path or os.environ.get("RASE_POLICY_PATH")
    if catalog is None:
        raise CollapseError(
            "catalog path required via --catalog, LIBERO_PLUS_TASK_CATALOG, "
            "or LIBERO_PLUS_ROOT"
        )
    if output is None:
        raise CollapseError("output path required via --output-dir or RASE_COLLAPSE_OUTPUT")
    if not args.dry_run and not policy:
        raise CollapseError("policy path required via --policy-path or RASE_POLICY_PATH")

    # Keep provenance JSON-stable (list, not tuple) so resume matches manifest.
    dimensions_list = (
        [str(item).strip() for item in dimensions] if dimensions is not None else None
    )
    levels_list = [int(level) for level in levels] if levels is not None else None
    suites_list = (
        [str(item).strip() for item in suites] if suites is not None else None
    )

    resolved = {
        "profile": profile_name,
        "selection": {
            "dimensions": dimensions_list,
            "levels": levels_list,
            "suites": suites_list,
            "smoke_tasks_per_cell": int(profile.get("tasks_per_cell", 1)),
        },
        "evaluation": {
            **evaluation,
            "episodes_per_task": episodes,
            "seed": args.seed if args.seed is not None else evaluation.get("seed", 0),
        },
        "catalog": str(catalog.resolve()),
        "output_dir": str(output.resolve()),
        "policy_path": str(Path(policy).expanduser().resolve()) if policy else None,
        "backend": args.backend_hook or args.backend,
    }
    return resolved, catalog, output, env_lock


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config, catalog_path, output_dir, env_lock = resolve_config(args)
        catalog = LiberoPlusTaskCatalog.load(catalog_path)
        selection = config["selection"]
        tasks = catalog.select(
            dimensions=selection["dimensions"],
            levels=selection["levels"],
            suites=selection["suites"],
            profile=config["profile"],
            smoke_tasks_per_cell=selection["smoke_tasks_per_cell"],
        )
        if not tasks:
            raise CollapseError("filters selected zero tasks")
        provenance = collect_provenance(REPO_ROOT, env_lock, config)
        manifest = ResultManifest.open_or_create(
            output_dir / "manifest.json", tasks, provenance
        )
        summary = {
            "dry_run": args.dry_run,
            "profile": config["profile"],
            "selected_tasks": len(tasks),
            "pending_tasks": len(manifest.pending()),
            "manifest": str(manifest.path),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.dry_run:
            return 0

        # Plus path config must be active before any libero.benchmark import.
        ensure_libero_plus_paths()
        backend = _load_hook(args.backend_hook) if args.backend_hook else require_lerobot_backend()
        run_tasks(
            manifest,
            tasks,
            backend,
            output_dir,
            config,
            continue_on_error=args.continue_on_error,
            max_attempts=args.max_attempts,
        )
        return 0
    except (CollapseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
