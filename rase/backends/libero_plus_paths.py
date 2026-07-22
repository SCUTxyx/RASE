"""Point LIBERO path lookups at a LIBERO-Plus checkout without mutating ~/.libero."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import yaml

from rase.eval.collapse import CollapseError


def resolve_libero_plus_root(explicit: str | Path | None = None) -> Path:
    """Resolve the LIBERO-Plus repository root or its `libero/libero` package root."""
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if _as_benchmark_root(candidate) is None:
            raise CollapseError(f"LIBERO-Plus root is invalid: {candidate}")
        return candidate.resolve()

    candidates: list[Path] = []
    env_root = os.environ.get("LIBERO_PLUS_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    try:
        import libero.libero as libero_pkg

        package_root = Path(libero_pkg.__file__).resolve().parent
        # .../LIBERO-plus/libero/libero -> repo root two levels up
        candidates.append(package_root.parents[1])
        candidates.append(package_root)
    except Exception:
        pass

    for candidate in candidates:
        if _as_benchmark_root(candidate) is not None:
            return candidate.resolve()
    raise CollapseError(
        "LIBERO-Plus root not found. Set LIBERO_PLUS_ROOT to the checkout "
        "(commit 4976dc3) or pass plus_root explicitly."
    )


def _as_benchmark_root(path: Path) -> Path | None:
    path = path.expanduser()
    if (path / "bddl_files").is_dir() and (path / "init_files").is_dir():
        return path.resolve()
    nested = path / "libero" / "libero"
    if (nested / "bddl_files").is_dir() and (nested / "init_files").is_dir():
        return nested.resolve()
    return None


def build_libero_plus_path_dict(plus_root: str | Path | None = None) -> dict[str, str]:
    root = _as_benchmark_root(resolve_libero_plus_root(plus_root))
    if root is None:
        raise CollapseError("resolved LIBERO-Plus path is missing bddl_files/init_files")
    datasets = root.parent / "datasets"
    return {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(datasets),
        "assets": str(root / "assets"),
    }


def default_libero_plus_config_dir() -> Path:
    override = os.environ.get("LIBERO_CONFIG_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".libero_plus_rase").resolve()


def ensure_libero_plus_paths(plus_root: str | Path | None = None) -> dict[str, str]:
    """Write a Plus-pointing config and make `get_libero_path` use it.

    Uses `LIBERO_CONFIG_PATH` (default `~/.libero_plus_rase`) so the stock
    `~/.libero/config.yaml` used by clean LIBERO baselines is left alone.
    """
    paths = build_libero_plus_path_dict(plus_root)
    for key in ("bddl_files", "init_states", "assets"):
        if not Path(paths[key]).is_dir():
            raise CollapseError(f"LIBERO-Plus {key} directory missing: {paths[key]}")

    config_dir = default_libero_plus_config_dir()
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
        # Fresh processes will import libero after LIBERO_CONFIG_PATH is set.
        pass
    return paths
