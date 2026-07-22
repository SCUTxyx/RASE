"""Configuration, provenance, and resumable manifests for collapse evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from rase.envs.task_catalog import LiberoPlusTask


SCHEMA_VERSION = 1
# ``failed`` stays retryable (Python exceptions). ``skipped`` is terminal and used
# after repeated native crashes (SIGSEGV/SIGFPE/SIGILL) leave a task stuck.
TERMINAL_STATUSES = frozenset({"completed", "skipped"})
DEFAULT_MAX_ATTEMPTS = 2


class CollapseError(RuntimeError):
    pass


class EvaluationBackend(Protocol):
    def __call__(
        self, task: LiberoPlusTask, task_output_dir: Path, config: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo_root: str | Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_provenance(
    repo_root: str | Path,
    env_lock: str | Path | None,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "git_sha": git_sha(repo_root),
        "env_lock_sha256": sha256_file(env_lock),
        "resolved_config": dict(resolved_config),
        "recorded_at": utc_now(),
    }


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


@dataclass(frozen=True)
class TaskResult:
    task: dict[str, Any]
    status: str
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None


class ResultManifest:
    """One atomically persisted status record per selected upstream task."""

    def __init__(self, path: str | Path, data: dict[str, Any]):
        self.path = Path(path)
        self.data = data

    @classmethod
    def open_or_create(
        cls,
        path: str | Path,
        tasks: Sequence[LiberoPlusTask],
        provenance: Mapping[str, Any],
    ) -> "ResultManifest":
        destination = Path(path)
        task_keys = [task.key for task in tasks]
        if len(task_keys) != len(set(task_keys)):
            raise CollapseError("selected tasks contain duplicate keys")
        if destination.exists():
            try:
                data = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CollapseError(f"cannot resume manifest {destination}: {exc}") from exc
            if data.get("schema_version") != SCHEMA_VERSION:
                raise CollapseError("manifest schema version is not supported")
            existing = list(data.get("task_order", []))
            if existing != task_keys:
                raise CollapseError(
                    "refusing to resume: selected task set/order differs from manifest"
                )
            manifest = cls(destination, data)
            # Round-trip through JSON so list/tuple and int/float don't block resume.
            old_provenance = json.loads(
                json.dumps(data.get("provenance", {}), sort_keys=True)
            )
            old_provenance.pop("recorded_at", None)
            new_provenance = json.loads(json.dumps(dict(provenance), sort_keys=True))
            new_provenance.pop("recorded_at", None)
            if old_provenance != new_provenance:
                untouched = all(
                    int(record.get("attempts", 0)) == 0
                    for record in data.get("results", {}).values()
                )
                if not untouched:
                    raise CollapseError(
                        "refusing to resume: provenance/config differs after execution began"
                    )
                # A dry-run manifest may be promoted to execution before any
                # task starts (for example, by adding the policy path).
                data["provenance"] = dict(provenance)
                manifest.save()
            return manifest

        results = {
            task.key: asdict(TaskResult(task=task.to_dict(), status="pending"))
            for task in tasks
        }
        data = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "provenance": dict(provenance),
            "task_order": task_keys,
            "results": results,
        }
        manifest = cls(destination, data)
        manifest.save()
        return manifest

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        atomic_write_json(self.path, self.data)

    def pending(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in self.data["task_order"]
            if self.data["results"][key]["status"] not in TERMINAL_STATUSES
        )

    def mark_running(self, key: str) -> None:
        record = self.data["results"][key]
        record.update(
            status="running",
            attempts=int(record.get("attempts", 0)) + 1,
            started_at=utc_now(),
            finished_at=None,
            error=None,
        )
        self.save()

    def mark_completed(self, key: str, metrics: Mapping[str, Any]) -> None:
        self.data["results"][key].update(
            status="completed",
            metrics=dict(metrics),
            finished_at=utc_now(),
            error=None,
        )
        self.save()

    def mark_failed(self, key: str, error: BaseException) -> None:
        self.data["results"][key].update(
            status="failed", finished_at=utc_now(), error=f"{type(error).__name__}: {error}"
        )
        self.save()

    def mark_skipped(self, key: str, reason: str) -> None:
        self.data["results"][key].update(
            status="skipped",
            finished_at=utc_now(),
            error=reason,
            metrics={
                "pc_success": None,
                "skipped": True,
                "skip_reason": reason,
            },
        )
        self.save()


def run_tasks(
    manifest: ResultManifest,
    tasks: Sequence[LiberoPlusTask],
    backend: EvaluationBackend,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    continue_on_error: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> None:
    if max_attempts < 1:
        raise CollapseError("max_attempts must be >= 1")
    by_key = {task.key: task for task in tasks}
    for key in manifest.pending():
        task = by_key[key]
        record = manifest.data["results"][key]
        prior_attempts = int(record.get("attempts", 0))
        # A native crash leaves status=running with attempts already counted.
        # After max_attempts, skip instead of looping forever on SIGSEGV/SIGILL.
        if prior_attempts >= max_attempts:
            manifest.mark_skipped(
                key,
                f"max_attempts={max_attempts} reached "
                f"(likely native crash on prior try; last_status={record.get('status')})",
            )
            continue
        task_dir = Path(output_dir) / "tasks" / task.key.replace(":", "__")
        task_dir.mkdir(parents=True, exist_ok=True)
        manifest.mark_running(key)
        try:
            metrics = backend(task, task_dir, config)
            if not isinstance(metrics, Mapping):
                raise CollapseError("backend must return a metrics mapping")
            manifest.mark_completed(key, metrics)
        except Exception as exc:
            manifest.mark_failed(key, exc)
            if not continue_on_error:
                raise


def require_lerobot_backend() -> Callable[..., Mapping[str, Any]]:
    """Return the built-in LeRobot + LIBERO-Plus task evaluation backend."""
    try:
        import lerobot  # noqa: F401
    except ImportError as exc:
        raise CollapseError(
            "LeRobot backend unavailable: install the pinned lerobot environment. "
            "Use --dry-run to validate catalog/config/manifest without a policy."
        ) from exc
    from rase.backends.lerobot_libero_plus import evaluate

    return evaluate

